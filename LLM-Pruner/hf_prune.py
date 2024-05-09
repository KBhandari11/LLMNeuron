import os
import sys
import random
import argparse

import csv 
import json

import torch
import numpy as np
import torch.nn as nn 
from LLMPruner.utils.logger import LoggerWithDepth
from compute_llama import prune_llama
from compute_bloom import prune_bloom
#from compute_single_dist import compute_single
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
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
    distribution_2 = []
    distribution_0 = []
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_values_F = []
        layer_values_2 = []
        layer_values_0 = []
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()
            layer_values_F.append(torch.linalg.matrix_norm(W).item()) #|W|_F norm
            layer_values_2.append(torch.linalg.matrix_norm(W, ord=2).item()) #|W|_2 norm
            layer_values_0.append((W.numel() - (W==0).sum().item())) #|W|_0 norm
        distribution_F.append(layer_values_F)
        distribution_2.append(layer_values_2)
        distribution_0.append(layer_values_0)
    return  np.array(distribution_F),np.array(distribution_2),np.array(distribution_0),float(count)/total_params

def initialize_distribution(all_distribution, keys):
    def add_nested_dict(dictionary, keys):
        current_dict = dictionary
        for key in keys:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        return all_distribution
    all_distribution = add_nested_dict(all_distribution, keys)
    return all_distribution

def set_random_seed(seed):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    set_random_seed(args.seed)
    with open("../dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_list = json.load(openfile)
    if not(args.save_distribution):
        logger = LoggerWithDepth(
                env_name="{}".format(args.save_ckpt_log_name), 
                config=args.__dict__,
                root_dir='prune_log',
                setup_sublogger=True,
                rank = (os.environ['RANK'])
            )
    else:
        logger = LoggerWithDepth(
                env_name="{}".format("./Here"), 
                config=args.__dict__,
                root_dir='prune_log',
                setup_sublogger=True
            )
    if  args.block_wise: 
        if args.pruner_type == "random":
            style = "block_random"
        else:
            style = "block"
    if  args.layer_wise: 
        style = "layer"
    if  args.channel_wise: 
        if args.pruner_type == "random":
            style = "channel_random"
        else:
            style = "channel"

    with open(args.save_distribution_path, 'r') as openfile:
        # Reading from json file
        all_distribution = json.load(openfile)
    args.save_model = False
    for i in range(5):
        print("Index", i)
        print("Sparsity", args.pruning_ratio*100, "%")
        print("Begin: Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024), file=sys.stderr)
        ratio = f"{int(args.pruning_ratio*100)}"
        keys = [str(i),style,ratio]
        all_distribution = initialize_distribution(all_distribution, keys)
        keys = [str(i),style,ratio]
        all_distribution = initialize_distribution(all_distribution, keys)
        if ("llama" in args.base_model) or ("vicuna" in args.base_model):
            distribution =prune_llama(logger,dataset_list,args)
        elif ("bloom" in args.base_model):
            distribution =prune_bloom(logger,dataset_list,args)
        all_distribution[str(i)][style][ratio].update(distribution)
        print("End: Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024), file=sys.stderr)
    
    if args.save_distribution:
        json_object = json.dumps(all_distribution, cls=NumpyEncoder)
        with open(args.save_distribution_path, "w") as outfile:
            outfile.write(json_object)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
    parser = argparse.ArgumentParser(description='Pruning (huggingface version)')
    
    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=2)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')

    parser.add_argument('--save_distribution', action='store_true', help='loop over multiple file')
    parser.add_argument('--save_distribution_path', type=str, help='path to save the distribution')

    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
