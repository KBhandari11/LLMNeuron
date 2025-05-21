import os
import sys
import random
import argparse
import pandas as pd

import csv 
import json
import re 

import torch
import numpy as np
import torch.nn as nn 
from collections import OrderedDict

import functools
from argparse import Namespace
from utils.dataset import getData
from utils.evaluation import evaluate, mcq_token_index,computeLogits
from transformers import AutoTokenizer, LlamaForCausalLM,LlamaTokenizerFast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.optim import AdamW,Adadelta
from utils.bag_of_words.projection_community import create_projection_network

from pynvml import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy
)

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['NCCL_P2P_DISABLE']='1'
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1236'+sys.argv[1]

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def free_mem(model):
    del_grad(model)
    del model
    torch.cuda.empty_cache()

def get_model(model_name,just_tokenizer=False):
    if model_name == "llama":
        base_model = "meta-llama/Llama-2-7b-hf"
    elif model_name == "llama_chat":
        base_model = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "vicuna":
        base_model = "lmsys/vicuna-7b-v1.5"
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if just_tokenizer:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        #low_cpu_mem_usage=True, 
        #device_map="auto"
    )
    #tokenizer.add_bos_token = False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #freezing the LM_HEAD
    for param in model.lm_head.parameters():
        param.requires_grad = False
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def print_without_rank(to_print, file=None,end=None):
    if os.environ.get('LOCAL_RANK',-1) == "0":
        if file != None:
            if end != None:
                print(to_print,flush=True)
            else:
                print(to_print, end=end,flush=True)
        else:
            if end != None:
                print(to_print, file=file,flush=True)
            else:
                print(to_print, file=file, end=end,flush=True)

def print_gpu_utilization():
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2} | Cached: {torch.cuda.memory_cached()/1024**2}")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l_0_norm(vector):
    count = 0
    total = 0
    for element in vector:
        for sub_element in element:
            if sub_element != 0:
                count += 1
    return count
def take_average(dict):
    data = dict["0"]
    iterations_block = list(["0","1","2","3","4"])
    iterations_channel = list(["0","1","2","3","4"])
    #for style , iterations in zip (["block","channel","block_random","channel_random"],[iterations_block,iterations_channel,iterations_block,iterations_channel]):
    for style , iterations in zip (["block","channel"],[iterations_block,iterations_channel,iterations_block,iterations_channel]):
        for iter in iterations:
            if iter == "0":
                continue
            for ratio in dict[iter][style]:
                for dataset in dict[iter][style][ratio]:
                    for norm in dict[iter][style][ratio][dataset]:
                        value = np.array(dict[iter][style][ratio][dataset][norm])
                        if len( value.shape) != 1:
                            shape_model = value.shape
                        data[style][ratio][dataset][norm]= (np.array(data[style][ratio][dataset][norm])+value)
                        if iter == iterations[-1]:
                            data[style][ratio][dataset][norm] = data[style][ratio][dataset][norm]/len(iterations)
    return data, shape_model
def strip(name):
    name = name.split("/")[-1]
    name = name.split("_")[0]
    return name 

def loop_over(dict):
    if isinstance(dict, list):
        print("end")
    else: 
        print(dict.keys())
        for keys in dict:
            loop_over(dict[keys])
        
def get_dataset_list(dataset_list):
    dataname = []
    for data in dataset_list:
        if "subset" not in dataset_list[data].keys():
            dataname.append(data)
        else:
            for subset in dataset_list[data]["subset"]:
                dataname.append(subset)
    return dataname

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def create_distribution_llm_pruner(model):
    layers = model.model.layers
    distribution_2 = []
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_values_2 = []
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()
            layer_values_2.append(torch.linalg.matrix_norm(W, ord=float("Inf")).item()) #|W|_inf norm
        distribution_2.append(layer_values_2)
    return  np.array(distribution_2)


def create_distribution_llm_pruner(model):
    layers = model.model.layers
    distribution_2 = []
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_values_2 = []
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()
            layer_values_2.append(torch.linalg.matrix_norm(W, ord=float("Inf")).item()) #|W|_inf norm
        distribution_2.append(layer_values_2)
    return  np.array(distribution_2)

 
def freeze_all_model(model):
    #all_self_modules = [f"{m.split('_')[1]}_proj" for m in modules_list]+[f"self_{m.split('_')[1]}_proj" for m in modules_list]
    modules=["attn.q", "attn.k", "attn.v", "attn.o","gate","mlp.up", "mlp.down"]
    all_modules = [f"{i}_{m}"  for i in range(3,31) for m in modules]
    layers = model.model.layers
    for idx_layer in range(len(layers)):
        layer = layers[idx_layer]
        for name1, child1 in layer.named_children():
            for name2, child2 in child1.named_children():
                if f"{idx_layer}_{name1.split('_')[-1]}.{name2.split('_')[0]}" in all_modules:
                    for param in child2.parameters():
                        param.requires_grad = True  
                else:
                    for param in child2.parameters():
                        param.requires_grad = False  
    return model

def freeze_subset_model(model, modules_list):
    #all_self_modules = [f"{m.split('_')[1]}_proj" for m in modules_list]+[f"self_{m.split('_')[1]}_proj" for m in modules_list]
    layers = model.model.layers
    for idx_layer in range(len(layers)):
        layer = layers[idx_layer]
        for name1, child1 in layer.named_children():
            for name2, child2 in child1.named_children():
                if f"{idx_layer}_{name1.split('_')[-1]}.{name2.split('_')[0]}" in modules_list:
                    for param in child2.parameters():
                        param.requires_grad = True  
                else:
                    for param in child2.parameters():
                        param.requires_grad = False  
    return model


def del_grad(model):
    #all_self_modules = [f"{m.split('_')[1]}_proj" for m in modules_list]+[f"self_{m.split('_')[1]}_proj" for m in modules_list]
    layers = model.model.layers
    for idx_layer in range(len(layers)):
        layer = layers[idx_layer]
        for name1, child1 in layer.named_children():
            for name2, child2 in child1.named_children():
                for param in child2.parameters():
                    if param.requires_grad:
                        del param.grad
    return model

def train_validation_dataloader(train_dataloader_list, validation_size):
    tensor_data = torch.stack([x.squeeze(0) for x,_,_,_ in train_dataloader_list[0:(-validation_size+1)]],dim=0) # transform to torch tensor
    tensor_atten = torch.stack([y.squeeze(0) for _,y,_,_ in train_dataloader_list[0:(-validation_size+1)]],dim=0) # transform to torch tensor
    tensor_label = torch.stack([z.squeeze(0) for _,_,z,_ in train_dataloader_list[0:(-validation_size+1)]],dim=0)
    training_label = torch.stack([aa.squeeze(0) for _,_,_,aa in train_dataloader_list[0:(-validation_size+1)]],dim=0)
    train_dataset = TensorDataset(tensor_data,tensor_atten,tensor_label,training_label) 

    val_tensor_data = torch.stack([x.squeeze(0) for x,_,_,_ in train_dataloader_list[(-validation_size+1):]],dim=0) # transform to torch tensor
    val_tensor_atten = torch.stack([y.squeeze(0) for _,y,_,_ in train_dataloader_list[(-validation_size+1):]],dim=0) # transform to torch tensor
    val_tensor_label = torch.stack([z.squeeze(0) for _,_,z,_ in train_dataloader_list[(-validation_size+1):]],dim=0)
    val_label = torch.stack([aa.squeeze(0) for _,_,_,aa in train_dataloader_list[(-validation_size+1):]],dim=0)
    val_dataset = TensorDataset(val_tensor_data,val_tensor_atten,val_tensor_label,val_label) 

    return train_dataset, val_dataset


# Function to check if a group contains all three patterns
def count_all_options(all_tokens):
    matches = []
    for i in range(len(all_tokens) - 3):
        group = all_tokens[i:i+3]  # Take three tokens
        if group[0] == '\n' and group[2] == '.':  # Check the first and third tokens
            matches.append(group)
    return len(matches)

def test_multigpu(model,tokenizer,rank, world_size, test_loader):
    model.eval()
    vocab_map = {v: k for k, v in tokenizer.get_vocab().items()} 
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data,atten, target,label in test_loader:
            data,atten, target = data.to(rank), atten.to(rank), target.to(rank)
            outputs = model(input_ids=data,attention_mask=atten,labels=target)
            #print(rank,"data: ",data.shape,tokenizer.batch_decode(data[0]))
            choices_index = mcq_token_index(tokenizer, [u for u in range(count_all_options(tokenizer.batch_decode(data[0])))])
            prediction =  [computeLogits(outputs.logits[idx,:],vocab_map,choices_index)[0] for idx in range(outputs.logits.shape[0])]# get the index of the max log-probability
            #print(rank,"Prediction",prediction)
            '''outputs = model(input_ids=data, attention_mask=atten, return_dict=True)
            logits = outputs.logits
            
            possible_token_ids =  torch.tensor([token for _, token in mcq_token_index(tokenizer, [u for u in range(count_all_options(tokenizer.batch_decode(data[0])))]).items()], device=rank)
            print(rank,mcq_token_index(tokenizer, [u for u in range(count_all_options(tokenizer.batch_decode(data[0])))]).items())
            logits_for_possible_tokens = F.softmax(logits, dim=-1)[:,-1, possible_token_ids]
            predicted_tokens = possible_token_ids[torch.argmax(logits_for_possible_tokens, dim=-1)]
            mcq_indices = target[:, 1]'''
            #print(count_all_options(tokenizer.batch_decode(data[0])),tokenizer.batch_decode(data[0]),flush=True) 
            #print(rank,'Data'," ".join(tokenizer.batch_decode(data[0])).replace("\n"," "))
            #print(rank,0," ".join(tokenizer.batch_decode(torch.argmax(logits[0,:,:], dim=-1))).replace("\n"," "))
            #print(rank, logits[:,-1, possible_token_ids].tolist(),flush=True)
            mcq_indices = label[:, 1]
            accuracy = sum([1 for predict_ans, target_ans in zip(prediction,tokenizer.batch_decode(mcq_indices)) if(predict_ans.lower() == target_ans.lower())])
            ddp_loss[0] += outputs.loss.item()
            ddp_loss[1] += accuracy 
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    test_loss = ddp_loss[0] / ddp_loss[2]
    if rank == 0:
        print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]), ddp_loss[1] / ddp_loss[2]), file=sys.stderr)
    return test_loss, ddp_loss[1] / ddp_loss[2]
 
def train_multigpu(model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    
    for data,atten, target,label in train_loader:
        #data, target = batch[0].to(rank), batch[1].to(rank)
        data,atten, target = data.to(rank), atten.to(rank), target.to(rank)
        optimizer.zero_grad()
        #print(rank, data.shape,atten.shape,target.shape,flush=True)
        outputs = model(input_ids=data,attention_mask=atten,labels=target)
        loss = outputs.loss
        #print(rank, loss, flush=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    model_size = compute_l2_norm(model) 
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f} \tModel Size: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1], model_size), file=sys.stderr)
    return ddp_loss[0] / ddp_loss[1], compute_l2_norm(model) 

def compute_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_norm += torch.norm(param).item()
    return l2_norm
def fsdp_main(rank, world_size, dataset,val_dataset, model, tokenizer, total_epoch,evaluation_parameters,save_result_parameters,save_checkpoint,lr=1e-5):
    setup(rank, world_size)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)#Adadelta

    sampler1 = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=False)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size, shuffle=False)

    train_kwargs = {'batch_size': 2, 'sampler': sampler1,"drop_last":True}
    test_kwargs = {'batch_size': 2, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(dataset,**train_kwargs)
    val_loader = DataLoader(val_dataset,**test_kwargs)
    
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer}
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    torch.cuda.set_device(rank)
    bfSixteen = MixedPrecision( param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    fpSixteen = MixedPrecision( param_dtype=torch.float16,reduce_dtype=torch.float16,buffer_dtype=torch.float16)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = FSDP(model,
                    auto_wrap_policy=llama_auto_wrap_policy,
                    mixed_precision=bfSixteen, 
                    cpu_offload=CPUOffload(offload_params=True),
                    sharding_strategy=sharding_strategy,
                    backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
                    use_orig_params=True,
                    device_id=torch.cuda.current_device())

    #scheduler = StepLR(optimizer, step_size=epoch)
    init_start_event.record()
    loss_list = []
    magnitude_list = []
    val_loss_list = []
    val_accuracy_list = []
    train_logger = []
    start_epoch = 1
    # Check if the logger CSV exists
    checkpoint_path =f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/checkpoint/temp/{save_checkpoint}.pt" 
    logger_csv_path = f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/checkpoint/temp/{save_checkpoint}.csv"
    if os.path.exists(logger_csv_path):
        print_without_rank(f"Logger CSV found. Loading existing data...", file=sys.stderr)
        df_logger = pd.read_csv(logger_csv_path)
        train_logger = df_logger.to_dict(orient="records")
        completed_epochs = df_logger['epoch'].max()
        start_epoch = completed_epochs + 1
        print_without_rank(f"Training has already completed {completed_epochs} epochs. Resuming from epoch {start_epoch}.", file=sys.stderr)
        
        # Restore lists based on the logger
        loss_list = df_logger['loss'].tolist()
        magnitude_list = df_logger['magnitude'].tolist()
        val_loss_list = df_logger['validation loss'].tolist()
        val_accuracy_list = df_logger['validation accuracy'].tolist()
        
        # Load the model checkpoint
        if os.path.exists(checkpoint_path):
            print_without_rank(f"Loading model checkpoint...", file=sys.stderr)
            checkpoint = torch.load(checkpoint_path)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)):
                model.load_state_dict(checkpoint)
        else:
            print_without_rank("Checkpoint file not found. Starting fresh training.", file=sys.stderr)
            loss_list = []
            magnitude_list = []
            val_loss_list = []
            val_accuracy_list = []
            train_logger = []

    # Training loop
    for epoch in range(start_epoch, total_epoch + 1):
        loss,magnitude = train_multigpu(model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        val_loss,val_accuracy = test_multigpu(model,tokenizer, rank, world_size, val_loader)
        loss_list.append(loss.item())
        magnitude_list.append(magnitude)
        val_loss_list.append(val_loss.item())
        val_accuracy_list.append(val_accuracy.item())  
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state = model.state_dict()
        if rank == 0:
            torch.save(cpu_state,checkpoint_path)
            train_logger.append({"epoch":epoch,
                                "loss":loss.item(),
                                "validation loss": val_loss.item(),
                                "magnitude":magnitude,
                                "validation accuracy":val_accuracy.item()
                                })
            df = pd.DataFrame(train_logger) 
            df.to_csv(logger_csv_path, index=False)

        #scheduler.step()
    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec", file=sys.stderr)
    # use a barrier to make sure training is done on all ranks
    dist.barrier()
    init_start_event.record()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type( model, StateDictType.FULL_STATE_DICT, save_policy):
        parameter_state = model.state_dict()
    init_end_event.record()
    #Evaluate the model here.
    
    if rank == 0:
        test_dataset,evaluation_args = evaluation_parameters
        free_mem(model)
        model, tokenizer = get_model(evaluation_args.base_model)
        model.to("cuda")
        model.load_state_dict(parameter_state)
        
        file, save_dictionary=save_result_parameters
        accuracy, accuracy_gen, predicted_actual,_ = evaluate(model=model,tokenizer=tokenizer,testloader=test_dataset,args=evaluation_args)
        print((accuracy[2],accuracy_gen[2]),(accuracy[0],accuracy_gen[0]),accuracy[1],loss_list,compute_l2_norm(model),flush=True)
        
        save_dictionary["loss"] = [str(l) for l in loss_list]
        save_dictionary["magnitude_list"] = [str(l) for l in magnitude_list]
        save_dictionary["val_loss_list"] = [str(l) for l in val_loss_list]
        save_dictionary["val_accuracy_list"] = [str(l) for l in val_accuracy_list]
        save_dictionary["final_model_l2"] = compute_l2_norm(model)

        save_dictionary['total'] = accuracy[1]
        save_dictionary['correct'] = accuracy[0]
        save_dictionary['test_accuracy'] = accuracy[2]
        save_dictionary['correct_gen'] = accuracy_gen[0]
        save_dictionary['test_accuracy_gen'] = accuracy_gen[2]
        save_dictionary['generated'] =predicted_actual
        save_dictionary['test_accuracy'] = accuracy[2]
        save_dictionary['test_accuracy_gen'] = accuracy_gen[2]
        save_result_csv(file,save_dictionary)
        states = model.state_dict()
        torch.save(states, f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/checkpoint/{save_checkpoint}.pt") 
    free_mem(model)
    cleanup()

def run_fsdp(dataset,val_dataset, model,tokenizer, epoch,evaluation_parameters,save_result_parameters,save_checkpoint, lr):
    WORLD_SIZE = torch.cuda.device_count()
    torch.cuda.empty_cache()
    mp.spawn(fsdp_main,
            args=(WORLD_SIZE,dataset,val_dataset,model,tokenizer, epoch,evaluation_parameters,save_result_parameters,save_checkpoint,lr),
            nprocs=WORLD_SIZE,
            join=True)

def save_result_csv(filename, data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writerow({key: data[key] for key in data.keys()})#this is redundant just to keep order consistent

def train_model_default(model,tokenizer, train_dataloader_list,validation_size, evaluation_parameters,save_result_parameters,save_checkpoint,num_epochs = 5, lr=1e-5):
    if len(train_dataloader_list) == 0:
        print("No data within dataset")
        print("No data within dataset",file=sys.stderr)
        return model 
    
    train_dataset,val_dataset = train_validation_dataloader(train_dataloader_list, validation_size)
    run_fsdp(train_dataset,val_dataset,model,tokenizer, num_epochs,evaluation_parameters,save_result_parameters,save_checkpoint,lr)



def get_high_datasets(ranked_dataset, top_skill= 50): 
    return ranked_dataset[:top_skill]
def freeze_and_train_model(model, modules_list,tokenizer, train_dataloader_list,validation_size,evaluation_parameters,save_result_parameters,save_checkpoint,num_epochs):
    model_new =  freeze_subset_model(model, modules_list)
    model_train  = train_model_default(model_new,tokenizer, train_dataloader_list, validation_size,evaluation_parameters,save_result_parameters,save_checkpoint,num_epochs,lr=1e-5)
    return model_train

def flatten_comprehension(matrix):
     return [item for row in matrix for item in row]


def get_all_dataset_list(dataset_info_list, dataset_list):
    dataname = []
    for d in dataset_list:
        for data in dataset_info_list:
            if "subset" not in dataset_info_list[data].keys():
                if  data == d:
                    dataname.append(data)
                    continue
            else:
                if d in dataset_info_list[data]["subset"]:
                    dataname.append([data,d])
                    continue
    return dataname

def get_modulesCommunityDataset(sparsity_ratio):
    with open("./dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_info_list = json.load(openfile)
    dataset_list = get_dataset_list(dataset_info_list)
    #Original Distribution
    with open("result/original_distribution_vicuna_7b.json", 'r') as openfile:
        vicuna_original = json.load(openfile)
    with open("result/original_distribution_llama_7b.json", 'r') as openfile:
        # Reading from json file
        llama_original = json.load(openfile)
    with open("result/original_distribution_llama_7b-chat.json", 'r') as openfile:
        # Reading from json file
        llama_chat_original = json.load(openfile)
    #Pruned Distribution
    with open("result/distribution_llama_7b.json", 'r') as openfile:
        # Reading from json file
        llama_distribution = json.load(openfile)
    with open("result/distribution_vicuna_7b.json", 'r') as openfile:
        # Reading from json file
        vicuna_distribution = json.load(openfile)
    with open("result/distribution_llama_7b-chat.json", 'r') as openfile:
        # Reading from json file
        llama_chat_distribution= json.load(openfile)
    with open("result/dataMultidisciplinaryCognitiveSkillsFrameworkRestrict.json", 'r') as openfile:
        dataCategory = json.load(openfile)

    llama_distribution, _ = take_average(llama_distribution)
    vicuna_distribution, _ = take_average(vicuna_distribution)
    llama_chat_distribution, _ = take_average(llama_chat_distribution)
    
    distribution_dist = [llama_distribution,llama_chat_distribution,vicuna_distribution]
    original_dist = [llama_original,llama_chat_original,vicuna_original]    
    modules_community_dataset = create_projection_network(dataCategory,dataset_list, distribution_dist, original_dist, sparsity_ratio = sparsity_ratio)
    return modules_community_dataset,dataset_info_list, dataset_list

def adjust_number(element,subset, all_elements, new_subset):
    parts = element.split('_')
    number_part = int(parts[0]) 
    rest = '_'.join(parts[1:])   
    count = 0
    while True:
        adjusted_number = random.choice(list(np.arange(number_part - 1-count, number_part + 1+count)))
        new_element = f'{adjusted_number}_{rest}'
        if new_element in all_elements and new_element not in new_subset and new_element not in subset:
            return new_element
        count +=1

def create_random_modules_set(all_modules, modules_list):
    new_subset = []
    for elem in modules_list:
        new_elem = adjust_number(elem,modules_list,all_modules, new_subset)
        new_subset.append(new_elem)
    return new_subset

def create_complete_random_modules_set(all_modules, modules_list, module_size=None):
    modules_set = set(modules_list)
    available_modules = list(set(all_modules) - modules_set)
    if not available_modules:
        raise ValueError("No modules available to create a random set different from the given modules_list.")
    random.shuffle(available_modules)
    if module_size ==None:
        module_size = len(modules_list)
    random_modules_set = available_modules[:module_size]
    return random_modules_set

def check_existing_runs(existing_runs,  model_name, comm_name, pruner_style, pruning_ratio, dataset, finetune):
    # Filter the DataFrame for rows that match the specified values
    existing_runs=existing_runs.drop_duplicates(subset=['iteration','model',"pruning_style","community","finetune","dataset"])
    condition = (
        (existing_runs['model'] == model_name) &
        (existing_runs['community'] == str(comm_name)) &
        (existing_runs['pruning_style'] == pruner_style) &
        (existing_runs['pruning_ratio'] == str(pruning_ratio)) &
        (existing_runs['dataset'] == dataset) &
        (existing_runs['finetune'] == finetune)
    )
    exists = not existing_runs[condition].empty
    if exists:
        filtered_df = existing_runs[
                (existing_runs['model'] == model_name) &
                (existing_runs['community'] == str(comm_name)) &
                (existing_runs['pruning_style'] == pruner_style) &
                (existing_runs['pruning_ratio'] == str(pruning_ratio)) &
                (existing_runs['dataset'] == dataset) &
                (existing_runs['finetune'] == finetune)][['accuracy_test', 'accuracy_test_gen']]
        return exists, filtered_df.iloc[0,:].tolist() 
    # Check if any row satisfies the condition
    return exists, None


if __name__ == "__main__":
    set_random_seed(int(sys.argv[2]))
    print([chr(65+i) for i in range(26)], file=sys.stderr)
    print("Cuda Version: ",torch.version.cuda, file=sys.stderr)
    sparsity_ratio = "20"
    validation_size = 100
    modules=["attn.q", "attn.k", "attn.v", "attn.o","gate","mlp.up", "mlp.down"]
    all_modules = [f"{i}_{m}"  for i in range(3,31) for m in modules]
    modules_community_dataset,dataset_info_list, dataset_list = get_modulesCommunityDataset(sparsity_ratio)
    existing_runs = pd.read_csv(f'./result/randomize_accuracy/finetuning_result_{sys.argv[2]}.csv',dtype=str)
    #"pruner_style","model","sparsity_ratio","community"
    for idx, model_name in enumerate(modules_community_dataset["model"]):
        pruner_style = modules_community_dataset["pruner_style"][idx]
        if str(sys.argv[3]) != model_name: 
            continue
        if str(sys.argv[4]) != pruner_style: 
            continue 
        community_data_lists = modules_community_dataset["community"]["kl"][idx]
        print("MODEL: ",model_name,sparsity_ratio,pruner_style, flush=True)
        for comm_name, community in community_data_lists.items(): 
            if int(comm_name) not in [0,1,2,3,4,5,6]:
                continue
            module_dataset_kl = get_high_datasets(modules_community_dataset["community"]["kl"][idx][comm_name]["dataset"]["all"], top_skill=5)
            module_dataset_info_format_kl = get_all_dataset_list(dataset_info_list, module_dataset_kl)
            rank_list = []
            count_dataset = 0
            for rank, (dataset_name_label,dataset_name)  in enumerate(zip(module_dataset_kl, module_dataset_info_format_kl)):
                if dataset_name_label in ["which_wiki_edit","authorship_verification"]:
                    continue
                args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=5000+validation_size,seqlen=1000,model_type="llama",num_process=10,max_length=10,device='cuda',fine_tune=False,evaluation_size=100, seed=0, base_model=model_name)
                tokenizer = get_model(model_name,just_tokenizer=True)
                train_dataset_list, test_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                '''if len(train_dataset_list)<1000:
                    continue
                ''' 
                module_accuracy_run = []
                for run_name in ["Community","All","Complete Random", "Complete Random w Community"]:
                    found_existing, filtered = check_existing_runs(existing_runs, model_name,comm_name, pruner_style, sparsity_ratio, dataset_name_label, run_name)
                    if found_existing:
                        print(dataset_name_label,run_name,filtered, " existing run found!", file=sys.stderr)
                        print(dataset_name_label,run_name,filtered, " existing run found!", flush=True)
                        continue
                    model, tokenizer = get_model(model_name)#nsamples=100,evaluation_size=50
                    print(dataset_name_label,run_name, file=sys.stderr)
                    community["modules"] = sorted(community["modules"])
                    save_dictionary = {"iteration":int(sys.argv[2]), 
                            "model":model_name, 
                            "pruning_style":pruner_style,
                            "community":comm_name,
                            "pruning_ratio":sparsity_ratio,
                            "dataset":dataset_name_label,
                            "finetune":run_name,
                            "modules_size":len(community["modules"]),
                            "modules":community["modules"],
                            "rank":rank+1,
                            "training_dataset_size":len(train_dataset_list)-validation_size,
                            "validation_test_size": validation_size
                            }
                    print([val for key, val in save_dictionary.items() if key != "modules"], flush=True, end="=> ")
                    evaluation_parameters = test_dataset,args_dataset
                    save_result_parameters = f'./result/randomize_accuracy/finetuning_result_{sys.argv[2]}.csv', save_dictionary
                    save_checkpoint = f"{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_{run_name}_{5000}_{5}"

                    #FINETUNE and EVALUATE                    
                    if run_name == "All":
                        model = freeze_all_model(model)
                        train_model_default(model,tokenizer, train_dataset_list,validation_size,evaluation_parameters,save_result_parameters,save_checkpoint, num_epochs = 5)
                    elif run_name == "Community":
                        module_list =  community["modules"]
                        freeze_and_train_model(model, module_list, tokenizer, train_dataset_list, validation_size,evaluation_parameters,save_result_parameters,save_checkpoint,num_epochs = 5)
                    elif run_name == "Random":
                        #module_type = [comm_m.split("_")[-1] for comm_m in community["modules"]]
                        #all_non_comm_module_list =  [m for m in all_modules if m not in community["modules"] and m.split("_")[-1] in module_type and m.split("_")[-1] in module_type] 
                        #module_list = random.sample(all_non_comm_module_list,len(community["modules"]))
                        module_list =  community["modules"]
                        new_module_list = create_random_modules_set(all_modules,module_list)  
                        save_dictionary["modules"] = new_module_list 
                        freeze_and_train_model(model, new_module_list, tokenizer, train_dataset_list,validation_size,evaluation_parameters,save_result_parameters,save_checkpoint,num_epochs = 5)
                    elif run_name == "Complete Random":
                        #module_type = [comm_m.split("_")[-1] for comm_m in community["modules"]]
                        #all_non_comm_module_list =  [m for m in all_modules if m not in community["modules"] and m.split("_")[-1] in module_type and m.split("_")[-1] in module_type] 
                        #module_list = random.sample(all_non_comm_module_list,len(community["modules"]))
                        module_list =  community["modules"]
                        new_module_list = create_complete_random_modules_set(all_modules,module_list)  
                        save_dictionary["modules"] = new_module_list 
                        freeze_and_train_model(model, new_module_list, tokenizer, train_dataset_list,validation_size,evaluation_parameters,save_result_parameters,save_checkpoint,num_epochs = 5)
                    elif run_name == "Complete Random w Community":
                        new_module_list = create_complete_random_modules_set(all_modules,[],module_size=len(community["modules"]))  
                        save_dictionary["modules"] = new_module_list 
                        freeze_and_train_model(model, new_module_list, tokenizer, train_dataset_list,validation_size,evaluation_parameters,save_result_parameters,save_checkpoint,num_epochs = 5)
                    del model
                if count_dataset >= 1:
                    break
                count_dataset += 1
            print("\n","*"*100)
        print("++"*100)