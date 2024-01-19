#CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=12 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes 1 --nproc_per_node 3 --master_port=25678 finetune_large.py> testing.txt 2>&1
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
    #FullStateDictConfig,
    #FullOptimStateDictConfig,
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
    
def get_gradient(rank, world_size, model, example_prompts):
    #model.train()
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
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD#SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    print(f"Rank: {rank} | PID: {os.getpid()}|", file=sys.stderr)
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
    print("Before Evaluation",file=sys.stderr)
    #print(torch.cuda.memory_summary(),file=sys.stderr)
    
    for idx, input in enumerate(train_loader):
        print("\t",rank,"Before Forward",file=sys.stderr)
        loss = model(input_ids=input["input_ids"].to(local_rank),labels=input["labels"].to(local_rank)).loss
        print("\t", rank,"Before Backward",file=sys.stderr)
        loss.backward()
        print("\t", idx, rank,"After Backward", file=sys.stderr)
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(input)
        print("\t", idx, rank,"Finished one iteration",file=sys.stderr)
    print("After Evaluation",file=sys.stderr)
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('\tLoss: {:.6f}'.format( fsdp_loss[0] / fsdp_loss[1]), file=sys.stderr)
    print("Completed ",file=sys.stderr)
    
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
    print("Completed ",rank, file=sys.stderr)
    return model
    
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
    with open("../dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_info_list = json.load(openfile)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dataset_list = get_dataset_list(dataset_info_list)
    dataset_name = "commonsense_qa"
    torch.cuda.empty_cache()
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-hf",
        low_cpu_mem_usage=True, #if args.torch_version >=1.9 else False,
    )
    model.to("cpu")
    model.train()
    model.zero_grad()
    print(f"Parent ID | PID: {os.getpid()}|", file=sys.stderr)
    args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=10,seqlen=400,model_type="llama",num_process=10,max_length=0,device='cpu',fine_tune=False)
    example_prompts, _ = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                        
    model = get_gradient(rank, world_size, model,example_prompts)
    dist.barrier()
    print("After Barrier ",file=sys.stderr)
    cleanup()

#CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=12 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes 1 --nproc_per_node 3 --master_port=25678 finetune_large.py > testing.txt 2>&1
