
import os
import gc
import sys
import functools
import copy
import json
from argparse import Namespace
from typing import Tuple
import torch.nn as nn 
import csv 
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP, LlamaDecoderLayer

#distributed FSDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float32,
)

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner

sys.path.append("../")
from utils.dataset import getData
from utils.evaluation import evaluate

def collate_fn(batch):
  return_data = {
      'input_ids': torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
      'labels': torch.stack([torch.tensor(x["labels"]) for x in batch])
    }
  return return_data


def convert_dataloader(lists,rank,world_size):
    my_dict = {}
    my_dict["input_ids"]= [x[0] for x,_ in lists]
    my_dict["labels"]= [y[0] for _,y in lists]
    dataset = Dataset.from_dict(my_dict)
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    setup()
    kwargs = {'batch_size': len(lists),'collate_fn':collate_fn, 'sampler': sampler,'num_workers': 2,'pin_memory': True,'shuffle': False}
    dataloader= DataLoader(dataset,**kwargs)
    return dataloader

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
    distribution_F = []
    distribution_0 = []
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_values_F = []
        layer_values_0 = []
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()
            layer_values_F.append(torch.linalg.matrix_norm(W).item()) #|W|_F norm
            layer_values_0.append((W.numel() - (W==0).sum().item())) #|W|_0 norm
        distribution_F.append(layer_values_F)
        distribution_0.append(layer_values_0)
    return float(count)/total_params, np.array(distribution_F),np.array(distribution_0)

def get_dataset_list(dataset_list):
    dataname = []
    for data in dataset_list:
        if "subset" not in dataset_list[data].keys():
            dataname.append(data)
        else:
            for subset in dataset_list[data]["subset"]:
                dataname.append([data,subset])
    return dataname

def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()
    
def get_gradient(rank, world_size, example_prompts,save_file, args):
    #model.train()
    torch.cuda.empty_cache()
    local_rank = int(os.environ['LOCAL_RANK'])
    train_loader = convert_dataloader(example_prompts, rank, world_size)

    torch.cuda.set_device(local_rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            #LlamaDecoderLayer, LlamaAttention, LlamaMLP,
            #LlamaDecoderLayer, LlamaAttention
            #LlamaMLP, LlamaAttention
            LlamaDecoderLayer
        },
    )
    model = LlamaForCausalLM.from_pretrained(
                args.base_model,
                low_cpu_mem_usage=True, #if args.torch_version >=1.9 else False,
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD#SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    # model is on CPU before input to FSDP
    model = FSDP(model,
        auto_wrap_policy=llama_auto_wrap_policy,
        mixed_precision=bfSixteen,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        cpu_offload=CPUOffload(offload_params=True),
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
        sync_module_states=True)
    #model._fsdp_wrap = True
    '''non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=True,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, LlamaMLP) #and  isinstance(submodule, LlamaAttention)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )'''
    fsdp_loss = torch.zeros(2).to(local_rank)
    #print(torch.cuda.memory_summary(),file=sys.stderr)
    
    for idx, input in enumerate(train_loader):
        loss = model(input_ids=input["input_ids"].to(local_rank),labels=input["labels"].to(local_rank)).loss
        loss.backward()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(input)
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('\tLoss: {:.6f}'.format( fsdp_loss[0] / fsdp_loss[1]), file=sys.stderr)
    
    #model = model.to(args.device)
    #save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    '''with FSDP.summon _full_params(
            model, StateDictType.FULL_STATE_DICT, save_policy,offload_to_cpu=True,
        ):
            model_dict = model.state_dict()
    if rank == 0:
        for para in model.parameters():
            print(para.grad[0], file=sys.stderr)'''
    '''if rank == 0:
        saved_gradients = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                saved_gradients[name] = param.grad.numpy()
        with open(save_file, 'w') as fp:
            json.dump(saved_gradients, fp, cls=NumpyEncoder)'''
        #torch.save(model_dict, temp_save_name)
    return model
    
def compute_single(logger,dataset_info_list,args): 
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    all_distribution = {"|W|_F":{},"|W|_0":{}}
    dataset_list = get_dataset_list(dataset_info_list)
    for dataset_name in dataset_list:
            logger.log("\n"+"*"*100+"*"*100+"\n")
            if isinstance(dataset_name,list):
                logger.log("DATASET: "+ " ".join(dataset_name))
            else:
                logger.log("DATASET: "+dataset_name)
            torch.cuda.empty_cache()
            tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    args.base_model,
                    low_cpu_mem_usage=True, #if args.torch_version >=1.9 else False,
                )
                if args.device != "cpu":
                    model.half()
                model.to(args.device)
                

                for param in model.parameters():
                    param.requires_grad_(True)
                before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                forward_prompts = torch.tensor([
                    [    1,   306,  4658,   278,  6593,   310,  2834,   338],
                    [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
                ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.
                #.to("cpu")
            pruner_type = args.pruner_type.lower()
            assert pruner_type in ['random', 'l2', 'l1', 'taylor']
            if pruner_type == 'random':
                imp = tp.importance.RandomImportance()
            elif pruner_type == 'l1':
                imp = llama_pruner.MagnitudeImportance(p=1)
            elif pruner_type == 'l2':
                imp = llama_pruner.MagnitudeImportance(p=2)
            elif pruner_type == 'taylor':
                imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
            else:
                raise NotImplementedError
            logger.log("Use {} pruner...".format(pruner_type))
            if args.block_wise:
                if rank ==0:
                    kwargs = {"importance": imp,
                        "global_pruning": args.global_pruning,
                        "iterative_steps": args.iterative_steps,
                        "ch_sparsity": args.pruning_ratio, 
                        "ignored_layers":[],
                        "channel_groups": {
                        },
                        "consecutive_groups": {
                            layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers #adding module for dataparallel
                        },
                        "customized_pruners": {
                            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                        },
                        "root_module_types": None, 
                        "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +  #adding module for dataparallel
                                        [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
                    }
                    logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
                    logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))
                    pruner = tp.pruner.MetaPruner(
                        model,
                        forward_prompts,
                        **kwargs
                        )
                    #setup()
                    model.train()
                    model.zero_grad()
                    logger.log("Start Pruning")

                for i in range(args.iterative_steps):
                    if pruner_type in ['taylor']:
                        args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=400,model_type="llama",num_process=10,max_length=0,device='cpu',fine_tune=False)
                        example_prompts, _ = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                        
                        #example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
                        logger.log("Start Backwarding in iterative steps = {}...".format(i))
                        
                        if args.taylor in ['param_mix', 'param_second']:
                            for input in example_prompts:
                                loss = model(input_ids=input["input_ids"].to(args.device),labels=input["labels"].to(args.device)).loss
                                loss.backward()
                                for module_param in model.parameters():
                                    module_param.grad = module_param.grad * module_param.grad / args.num_examples
                                    if hasattr(module_param, 'acc_grad'):
                                        module_param.acc_grad += module_param.grad
                                    else:
                                        module_param.acc_grad = copy.deepcopy(module_param.grad)
                                model.zero_grad()
                                del loss.grad
                        #print(f"Parent ID | rank: {rank} | PID: {os.getpid()}|", file=sys.stderr)
                        modelGradient = get_gradient(rank, world_size,example_prompts,"temp/test.json",args)
                        logger.log("Finished collecting gradient")
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.summon_full_params(
                        modelGradient, StateDictType.FULL_STATE_DICT, save_policy,offload_to_cpu=True,
                    ):
                        cpu_state = modelGradient.state_dict()
                    dist.barrier()
                    cleanup()
                    if rank==0:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                param.grad = copy.deepcopy(cpu_state[name])
                    del cpu_state
                    del modelGradient
                    if rank == 0:
                        pruner.step()
                        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        
                        # modify inferece-related attributes
                        for layer in model.model.layers:
                            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
                if rank == 0:
                    # Clean the gradient in the model
                    model.zero_grad()
                    for name, module in model.named_parameters():
                        if 'weight' in name:
                            module.grad = None
                    #if rank == 0:
                    del pruner
                
            elif args.channel_wise:
                kwargs = {
                    "importance": imp,
                    "global_pruning": args.global_pruning,
                    "iterative_steps": args.iterative_steps,
                    "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
                    "ignored_layers":[],
                    #"round_to": model.config.num_attention_heads * 2,
                    "channel_groups": {
                        #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
                    },
                    "customized_pruners": {
                        LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                        #LlamaAttention: llama_pruner.hf_attention_pruner,
                    },
                    "root_module_types": [LlamaRMSNorm, LlamaAttention],
                }
                pruner = tp.pruner.MetaPruner(
                    model,
                    forward_prompts,
                    **kwargs
                )
                model.zero_grad()
            
                logger.log("Start Pruning")
                for i in range(args.iterative_steps):
                    if pruner_type in ['taylor']:
                        args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=10,seqlen=400,model_type="llama",num_process=10,max_length=0,device='cpu',fine_tune=False)
                        example_prompts, _ = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                        #example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
                        logger.log("Start Backwarding in iterative steps = {}...".format(i))
                        loss_value = 0
                        for train_data ,label_data in example_prompts:
                            input = train_data.to(args.device)
                            label = label_data.to(args.device)
                            loss = model(input, labels=label).loss
                            loss.backward()
                            loss_value += loss.item()
                            del(input)
                            del(label)
                        loss = loss_value /len(example_prompts)
                        logger.log("Loss = {}".format(loss))
                    pruner.step()

                    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

                # Clean the gradient in the model
                model.zero_grad()
                for name, module in model.named_parameters():
                    if 'weight' in name:
                        module.grad = None

                # modify inferece-related attributes
                model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
                model.zero_grad()
                del pruner          
            elif args.layer_wise:
                model.model.layers = model.model.layers[:args.layer]
                after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            else:
                raise NotImplementedError
            if rank ==0 :
                logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
                gc.collect()
            torch.cuda.empty_cache()
            if args.save_model:
                #model.half()
                if "chat" in args.base_model.split("-"):
                    isChat = "-chat"
                else:
                    isChat = ""
                if args.block_wise:
                    print("Saving Block Wise")
                    pruneType = 'block'
                    torch.save(model.state_dict(), f"/data/Kushal/MLNeuron/checkpoints/llama-7b/LLM_Pruner/block/{int(args.pruning_ratio*100)}{isChat}/model_{dataset_name[-1].split('/')[-1]}.pt")
                elif args.layer_wise:
                    print("Saving Layer Wise")
                    pruneType = 'layer'
                    torch.save(model.state_dict(), f"/data/Kushal/MLNeuron/checkpoints/llama-7b/LLM_Pruner/layer/{int(args.pruning_ratio*100)}{isChat}/model_{dataset_name[-1].split('/')[-1]}.pt")
                elif args.channel_wise:
                    print("Saving Channel Wise")
                    pruneType = 'channel'
                    torch.save(model.state_dict(), f"/data/Kushal/MLNeuron/checkpoints/llama-7b/LLM_Pruner/channel/{int(args.pruning_ratio*100)}{isChat}/model_{dataset_name[-1].split('/')[-1]}.pt")
                '''torch.save({
                    'model': model, 
                    'tokenizer': tokenizer,
                }, logger.best_checkpoint_path)'''
            if rank ==0 :
                if args.save_distribution:
                    if isinstance(dataset_name,list):
                        sparsity, all_distribution["|W|_F"][dataset_name[-1].split('/')[-1]] , all_distribution["|W|_0"][dataset_name[-1].split('/')[-1]]  = create_distribution_llm_pruner(model)
                        logger.log(f"{dataset_name[-1].split('/')[-1]}: Total Sparsity {sparsity}")
                    else:
                        sparsity, all_distribution["|W|_F"][dataset_name.split('/')[-1]] , all_distribution["|W|_0"][dataset_name.split('/')[-1]]  = create_distribution_llm_pruner(model)
                        logger.log(f"{dataset_name.split('/')[-1]}: Total Sparsity {sparsity}")

                torch.cuda.empty_cache()

            logger.log("\n==================Finish================\n")
            logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
            if rank ==0:
                del model
    if args.save_distribution:
        return all_distribution