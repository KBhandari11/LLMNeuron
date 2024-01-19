import os
import sys
import random
import argparse

import csv 
import json

import torch
import numpy as np
from LLMPruner.utils.logger import LoggerWithDepth
from compute_both import compute_both
#from compute_single import compute_single
from compute_single_dist import compute_single
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
def create_distribution_llm_pruner(model):
    distribution_F = []
    distribution_0 = []
    count = 0 
    total_params = 0 
    for layers in range(32):
        data_layer_F = []
        data_layer_0 = []
        for sub_layer in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            key_name = f'model.layers.{layers}.{sub_layer}.weight'
            if key_name in model.keys():
                W = model[key_name]
                count += (W==0).sum().item()
                total_params += W.numel()
                data_layer_F.append(torch.linalg.matrix_norm(W).item())#|W|_F norm
                data_layer_0.append((W.numel() - (W==0).sum().item()))#|W|_0 norm
            else:
                data_layer_F.append(0)
                data_layer_0.append(0)
        distribution_F.append(data_layer_F)
        distribution_0.append(data_layer_0)
    return float(count)/total_params, np.array(distribution_F),np.array(distribution_0)

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
    dataset_both = [["commonsense_qa","tasksource/mmlu"],["commonsense_qa","math_qa"],["commonsense_qa","EleutherAI/truthful_qa_mc"],["commonsense_qa","derek-thomas/ScienceQA"],["tasksource/mmlu","derek-thomas/ScienceQA"],["math_qa","derek-thomas/ScienceQA"]]
    if not(args.save_distribution):
        logger = LoggerWithDepth(
                env_name="{}".format(args.save_ckpt_log_name), 
                config=args.__dict__,
                root_dir='prune_log',
                setup_sublogger=True
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
    if args.save_distribution:
        with open(args.save_distribution_path, 'r') as openfile:
            # Reading from json file
            all_distribution = json.load(openfile)
        args.save_model = False
        if args.do_train_both:
            print("Both")
            for i in range(10):
                print("Index", i)
                isChat = "-chat" if "chat" in args.base_model.split("-") else ""
                ratio = f"{int(args.pruning_ratio*100)}{isChat}"
                keys = [str(i),"|W|_F",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                keys = [str(i),"|W|_0",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                distribution = compute_both(logger,dataset_both,dataset_list,args)
                w_f,w_0 =distribution["|W|_F"],distribution["|W|_0"]
                all_distribution[str(i)]["|W|_F"][style][ratio].update(w_f)
                all_distribution[str(i)]["|W|_0"][style][ratio].update(w_0)
        else:
            print("Single")
            for i in range(10):
                print("Index", i)
                isChat = "-chat" if "chat" in args.base_model.split("-") else ""
                ratio = f"{int(args.pruning_ratio*100)}{isChat}"
                keys = [str(i),"|W|_F",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                keys = [str(i),"|W|_0",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                distribution =compute_single(logger,dataset_list,args)
                w_f, w_0 =distribution["|W|_F"],distribution["|W|_0"]
                all_distribution[str(i)]["|W|_F"][style][ratio].update(w_f)
                all_distribution[str(i)]["|W|_0"][style][ratio].update(w_0)
        json_object = json.dumps(all_distribution, cls=NumpyEncoder)
        with open(args.save_distribution_path, "w") as outfile:
            outfile.write(json_object)
    else:
        if args.do_train_both:
            compute_both(logger,dataset_both,dataset_list,args)
        else:
            compute_single(logger,dataset_list,args)
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')
    
    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
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
    parser.add_argument('--do_train_both', action='store_true', help='mix dataset for training')

    parser.add_argument('--save_distribution', action='store_true', help='loop over multiple file')
    parser.add_argument('--save_distribution_path', type=str, help='path to save the distribution')

    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
