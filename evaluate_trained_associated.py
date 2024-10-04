import os
import sys
import random
import argparse
import pandas as pd

import csv 
import json

import torch
import numpy as np
import torch.nn as nn 
from collections import OrderedDict

import functools
from argparse import Namespace
from utils.dataset import getData
from utils.evaluation import evaluate
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.optim import AdamW,Adadelta
from utils.bag_of_words.projection_community import *

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
)


os.environ['NCCL_P2P_DISABLE']='1'
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def print_without_rank(to_print, file=None,end=None):
    if os.environ.get('LOCAL_RANK',-1) == "0":
        if file != None:
            if end != None:
                print(to_print)
            else:
                print(to_print, end=end)
        else:
            if end != None:
                print(to_print, file=file)
            else:
                print(to_print, file=file, end=end)

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


def get_model(model_name):
    if model_name == "llama":
        base_model = "meta-llama/Llama-2-7b-hf"
    elif model_name == "llama_chat":
        base_model = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "vicuna":
        base_model = "lmsys/vicuna-7b-v1.5"
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        #low_cpu_mem_usage=True, 
        #device_map="auto"
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

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

def freeze_model(model, modules_list):
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

class DataDataset(torch.utils.data.Dataset):
    def __init__(self, train_dataloader_list):
        self.input = [x.squeeze(0) for x,_,_ in train_dataloader_list]
        self.attention = [y.squeeze(0) for _,y,_ in train_dataloader_list]
        self.labels = [z.squeeze(0) for _,_,z in train_dataloader_list]

    def __getitem__(self, idx):
        item = {"input_ids": torch.Tensor(self.input[idx])}
        item['attention_mask'] = torch.Tensor(self.attention[idx])
        item['labels'] = torch.Tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_norm += torch.norm(param).item()
    return l2_norm

def train_multigpu(model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    
    for data,atten, target in train_loader:
        #data, target = batch[0].to(rank), batch[1].to(rank)
        data,atten, target = data.to(rank), atten.to(rank), target.to(rank)
        optimizer.zero_grad()
        outputs = model(input_ids=data,attention_mask=atten,labels=target)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]), file=sys.stderr)

def fsdp_main(rank, world_size ,dataset, model, epoch,lr=1e-5):
    setup(rank, world_size)

    optimizer = AdamW(model.parameters(), lr=lr,weight_decay=0.0)#Adadelta

    sampler1 = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=False)

    train_kwargs = {'batch_size': 2, 'sampler': sampler1,"drop_last":True}
    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(dataset,**train_kwargs)
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
                  #mixed_precision=fpSixteen, 
                  cpu_offload=CPUOffload(offload_params=True),
                  sharding_strategy=sharding_strategy,
                  backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
                  use_orig_params=True,
                  device_id=torch.cuda.current_device())

    #scheduler = StepLR(optimizer, step_size=epoch)
    init_start_event.record()
    for epoch in range(1, epoch + 1):
        train_multigpu(model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        #scheduler.step()
    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec", file=sys.stderr)
    # use a barrier to make sure training is done on all ranks
    dist.barrier()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type( model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if rank == 0: 
        torch.save(cpu_state, "./temp/test.pt")
    cleanup()
    del model

def run_fsdp(dataset, model, epoch, lr):
    WORLD_SIZE = torch.cuda.device_count()
    torch.cuda.empty_cache()
    mp.spawn(fsdp_main,
            args=(WORLD_SIZE, dataset,model, epoch,lr),
            nprocs=WORLD_SIZE,
            join=True)

def train_model_default(train_dataloader_list,model,num_epochs = 3, lr=1e-5):
    if len(train_dataloader_list) == 0:
        print("No data within dataset")
        print("No data within dataset",file=sys.stderr)
        return model 
    tensor_data = torch.stack([x.squeeze(0) for x,_,_ in train_dataloader_list],dim=0) # transform to torch tensor
    tensor_atten = torch.stack([y.squeeze(0) for _,y,_ in train_dataloader_list],dim=0) # transform to torch tensor
    tensor_label = torch.stack([z.squeeze(0) for _,_,z in train_dataloader_list],dim=0)
    #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = TensorDataset(tensor_data,tensor_atten,tensor_label) # create your datset
    #train_dataset = DataDataset(train_dataloader_list)

    #accelerator.print(f"{AcceleratorState()}")
    run_fsdp(train_dataset,model,num_epochs,lr)
    state_dict = torch.load( "./temp/test.pt")
    model.load_state_dict(state_dict)
    return model 


def get_high_datasets(ranked_dataset, top_skill= 50): 
    return ranked_dataset[:top_skill]

def freeze_and_train_model(model, modules_list,train_dataloader_list,num_epochs):
    model_new =  freeze_model(model, modules_list)
    model_train  = train_model_default(train_dataloader_list,model_new,num_epochs,lr=1e-4)
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
    modules_community_dataset = create_projection_network(dataCategory,dataset_list, distribution_dist, original_dist, sparsity_ratio = sparsity_ratio, random_seed=True)
    return modules_community_dataset,dataset_info_list, dataset_list

def free_mem(model):
    del_grad(model)
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    set_random_seed(int(sys.argv[1]))
    sparsity_ratio = "20"
    modules=["attn.q", "attn.k", "attn.v", "attn.o"]
    all_modules = [f"{i}_{m}"  for i in range(3,31) for m in modules]
    data = {"iteration":[],"model":[],"pruning_style":[],"community":[],"pruning_ratio":[],"dataset":[],"accuracy":[],"rank":[],"modules":[],"modules_size":[],"finetune":[],"l2":[]}
    modules_community_dataset,dataset_info_list, dataset_list = get_modulesCommunityDataset(sparsity_ratio)
    print("dataset_name,run_name,community,pruner_style,accuracy,modules_size,l2")
    #"pruner_style","model","sparsity_ratio","community"
    for idx, model_name in enumerate(modules_community_dataset["model"]):
        #print(idx, model_name, modules_community_dataset["pruner_style"][idx])
        #Just working with KL divergence based approach
        community_data_lists = modules_community_dataset["community"]["kl"][idx]
        for comm_name, community in community_data_lists.items():
            #print("Community Name:",comm_name)
            #get_high_datasets(modules_community_dataset["community"]["network"][idx][comm_name]["dataset"], top_skill=20)
            module_dataset_kl = get_high_datasets(modules_community_dataset["community"]["kl"][idx][comm_name]["dataset"]["all"], top_skill=10)
            module_dataset_info_format_kl = get_all_dataset_list(dataset_info_list, module_dataset_kl)
            #Just working with KL divergence based approach,
            module_accuracy = []
            rank_list = []
            rank = 1
            for dataset_name_label,dataset_name  in zip(module_dataset_kl, module_dataset_info_format_kl):
                module_accuracy_run = []
                for run_name in ["All","Community","Random"]:
                    model, tokenizer = get_model(model_name)
                    args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=100,seqlen=1000,model_type="llama",num_process=10,max_length=100,device='cuda',fine_tune=False,evaluation_size=50, seed=0)
                    train_dataset_list, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                    print(dataset_name_label,run_name, file=sys.stderr)
                    if run_name == "All":
                        finetuned_model = train_model_default(train_dataset_list,model,num_epochs = 1)
                        module_list = []
                    elif run_name == "Community":
                        module_list =  community["modules"]
                        finetuned_model = freeze_and_train_model(model, module_list, train_dataset_list,num_epochs = 1)
                    elif run_name == "Random":
                        all_non_comm_module_list =  [m for m in all_modules if m not in community["modules"] and m.split["_"][-1] not in [comm_m.split["_"][-1] for comm_m in community["modules"]]] 
                        module_list = random.sample(all_non_comm_module_list,len(community["modules"]))
                        finetuned_model = freeze_and_train_model(model, module_list, train_dataset_list,num_epochs = 1)
                        #rank = modules_community_dataset["community"]["kl"][idx][comm_name]["dataset"]["all"].index(dataset_name_label)
                    finetuned_model = finetuned_model.to("cuda")
                    accuracy, _ = evaluate(model=finetuned_model,tokenizer=tokenizer,testloader=validation_dataset,args=args_dataset)

                    module_accuracy_run.append(accuracy)
                    after = compute_l2_norm(finetuned_model)
                    free_mem(finetuned_model)
                    free_mem(model)
                    del finetuned_model
                    del model
                    
                    print(f"{dataset_name_label},{run_name},{comm_name},{modules_community_dataset['pruner_style'][idx]},{accuracy[2]},{len(module_list)},{after}")
                    data["iteration"].append(int(sys.argv[1]))
                    data["model"].append(model_name)
                    data["pruning_style"].append(modules_community_dataset["pruner_style"][idx])
                    data["community"].append(comm_name)
                    data["pruning_ratio"].append(sparsity_ratio)
                    data["dataset"].append(dataset_name_label)
                    data["accuracy"].append(accuracy[2])
                    data["rank"].append(rank)
                    data["modules"].append(module_list)
                    data["modules_size"].append(len(module_list))
                    data["finetune"].append(run_name)
                    data["l2"].append(after)
                module_accuracy.append(module_accuracy_run)
                rank_list.append(rank)
                rank +=1
            print("\n","*"*100)
            print(module_list)
            print("Module Accuracy",comm_name,module_accuracy, flush=True)
            print("\n","*"*100)

        print("++"*100)
        df = pd.DataFrame(data)
        df.to_csv(f'./result/randomize_accuracy/randomize_data_new_kl_{sys.argv[1]}.csv',index=False) 

    df = pd.DataFrame(data)
    df.to_csv(f'./result/randomize_accuracy/randomize_data_new_kl_{sys.argv[1]}.csv', index=False) 
