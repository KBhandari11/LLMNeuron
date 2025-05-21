import os
import sys
import random
import argparse
import pandas as pd

import csv 
import json
import re 
from ast import literal_eval

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
        print("\tend")
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


def compute_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_norm += torch.norm(param).item()
    return l2_norm

def get_high_datasets(ranked_dataset, top_skill= 50): 
    return ranked_dataset[:top_skill]
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
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import numpy as np


def extract_weights(state_dict, modules):
    weights = {}
    for module_name in modules:
        for key in state_dict.keys():
            if "layers" not in key:
                continue
            key_split= key.split(".")[2:5]
            if "gate" in key:
                format_module_key = f"{key_split[0]}_{key_split[2].split('_')[0]}"
            else:
                format_module_key = f"{key_split[0]}_{key_split[1].split('_')[-1]}.{key_split[2].split('_')[0]}"
            if module_name in format_module_key:  # Match affected module names
                weights[module_name] = state_dict[key].detach().cpu()
    del state_dict
    return weights

def compare_weights(base, fine_tuned):
    differences = {}
    for key in base.keys():
        if key in fine_tuned:
            base_param = base[key].numpy().flatten()
            fine_tuned_param = fine_tuned[key].numpy().flatten()
            #print("\tmean_absolute_difference", np.mean(np.abs(base_param - fine_tuned_param)))
            differences[key] = {
                "cosine_similarity": 1 - cosine(base_param, fine_tuned_param),
                "euclidean_distance": np.linalg.norm(base_param - fine_tuned_param),
                "mean_absolute_difference": np.mean(np.abs(base_param - fine_tuned_param))
            }
            '''print( {
                "cosine_similarity": 1 - cosine(base_param, fine_tuned_param),
                "euclidean_distance": np.linalg.norm(base_param - fine_tuned_param),
             #   "mean_absolute_difference": np.mean(np.abs(base_param - fine_tuned_param))
            })'''
    return differences

def plot_differences(differences, title, metric1, metric2=None, metric3=None):
    modules = list(differences.keys())
    if metric2 == None or metric3 == None:
        values = [differences[module][metric1] for module in modules]

        plt.figure(figsize=(12, 6))
        plt.bar(modules, values)
        plt.title(f"{title} - {metric1}")
        plt.xticks(rotation=90)
        plt.ylabel(metric1)
        plt.xlabel("Modules")
        plt.show()
    else:
        values_metric1 = [differences[module][metric1] for module in modules]
        values_metric2 = [differences[module][metric2] for module in modules]
        values_metric3 = [differences[module][metric3] for module in modules]

        fig, [ax1, ax2,ax3] = plt.subplots(figsize=(20, 6),ncols=3)
        ax1.bar(modules, values_metric1)
        ax1.set_title(f"{title} - {metric1}")
        ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
        ax1.set_ylabel(metric1)
        ax1.set_xlabel("Modules")

        ax2.bar(modules, values_metric2)
        ax2.set_title(f"{title} - {metric2}")
        ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
        ax2.set_ylabel(metric2)
        ax2.set_xlabel("Modules")

        ax3.bar(modules, values_metric3)
        ax3.set_title(f"{title} - {metric3}")
        ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
        ax3.set_ylabel(metric3)
        ax3.set_xlabel("Modules")
        plt.show()

if __name__ == "__main__":
    set_random_seed(int(2))
    sparsity_ratio = "20"
    modules=["attn.q", "attn.k", "attn.v", "attn.o","gate","mlp.up", "mlp.down"]
    all_modules = [f"{i}_{m}"  for i in range(3,31) for m in modules]
    model_path = "/gpfs/u/home/LLMG/LLMGbhnd/scratch/checkpoint/"
    #modules_community_dataset,dataset_info_list, dataset_list = get_modulesCommunityDataset(sparsity_ratio)
    data = pd.read_csv(f'./result/randomize_accuracy/finetuning_result_modified.csv',dtype=str).drop_duplicates(subset=['iteration','model',"pruning_style","community","finetune","dataset"])
    data["modules"] = data["modules"].apply(literal_eval)
    #pd.options.display.max_colwidth = 100
    pd.options.display.max_columns = 30
    grouped_dict = data.groupby(['model', 'pruning_style', 'community', 'pruning_ratio', 'dataset', "modules_size", "rank",  "training_dataset_size","validation_dataset_size" ])
    for idx, (gp_name,grouped) in enumerate(grouped_dict):
        if grouped['finetune'].unique().shape[0] != 5:
            continue
        print(gp_name)
        model_name = grouped["model"].iloc[0] 
        pruner_style = grouped["pruning_style"].iloc[0] 
        comm_name = grouped["community"].iloc[0] 
        dataset_name_label = grouped["dataset"].iloc[0]
        loss_training = literal_eval(grouped["loss_training"].iloc[0])[-1]
        loss_validation = literal_eval(grouped["loss_validation"].iloc[0])[-1]
        accuracy_validation =literal_eval(grouped["accuracy_validation"].iloc[0])[-1]
        accuracy = grouped["accuracy_test"].iloc[0]


        module_list = grouped.loc[grouped['finetune'] == 'Community', 'modules'].iloc[0] 
        random_module_list = grouped.loc[grouped['finetune'] == 'Complete Random', 'modules'].iloc[0] 
        comp_random_module_list = grouped.loc[grouped['finetune'] == 'Complete Random w Community', 'modules'].iloc[0] 


        base_model_data,_ = get_model(model_name)
        base_model = base_model_data.state_dict()
        base_weights = extract_weights(base_model, list(set(module_list+random_module_list+comp_random_module_list)))
        del base_model 

        '''all_cond = os.path.exists(f"{model_path}/{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_All_{5000}_{5}.pt")
        community_cond = os.path.exists(f"{model_path}/{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_Community_{5000}_{5}.pt")
        random_cond = os.path.exists(f"{model_path}/{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_Random_{5000}_{5}.pt")
        print((all_cond, community_cond, random_cond),(all_cond and community_cond and random_cond)) 
        if not(all_cond and community_cond and random_cond):
            continue'''
    
        fine_tuned_all = torch.load(f"{model_path}/{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_All_{5000}_{5}.pt", map_location=torch.device("cpu"))
        all_weights_subset = extract_weights(fine_tuned_all, module_list)
        all_weights_all = extract_weights(fine_tuned_all, all_modules)
        print("\tAll(subset)",all_weights_subset.keys())
        print("\tAll(all)",all_weights_all.keys())
        del fine_tuned_all 
        
        fine_tuned_community =torch.load(f"{model_path}/{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_Community_{5000}_{5}.pt", map_location=torch.device("cpu"))
        community_weights = extract_weights(fine_tuned_community, module_list)
        print("\tCommunity",community_weights.keys())
        del fine_tuned_community
        
        fine_tuned_random = torch.load(f"{model_path}/{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_Complete Random_{5000}_{5}.pt", map_location=torch.device("cpu"))
        random_weights = extract_weights(fine_tuned_random, random_module_list)
        print("\tRandom",random_weights.keys()) 
        del fine_tuned_random 

        fine_tuned_com_random = torch.load(f"{model_path}/{model_name}_{pruner_style}_{comm_name}_{dataset_name_label}_Complete Random w Community_{5000}_{5}.pt", map_location=torch.device("cpu"))
        random_weights_w_comm = extract_weights(fine_tuned_com_random, comp_random_module_list)
        print("\tRandom w Community",random_weights.keys()) 
        del fine_tuned_com_random 
        
        #Compare Weights
        diff_all_all = compare_weights(base_weights, all_weights_all)
        diff_all_subset = compare_weights(base_weights, all_weights_subset)
        diff_community = compare_weights(base_weights, community_weights)
        diff_random = compare_weights(base_weights, random_weights)
        diff_random_w_comm = compare_weights(base_weights, random_weights_w_comm)
        save_dictionary = {
                        "model":model_name, 
                        "pruning_style":pruner_style,
                        "community":comm_name,
                        "pruning_ratio":sparsity_ratio,
                        "dataset":dataset_name_label,
                        "modules":module_list,
                        "random_modules":random_module_list,
                        "random_w_community_modules":comp_random_module_list,
                        "diff_all_all":diff_all_all,
                        "diff_all_subset":diff_all_subset, 
                        "diff_community":diff_community,
                        "diff_random":diff_random,
                        "diff_random_w_comm":diff_random_w_comm,
                        "loss_training":loss_training,
                        "loss_validation":loss_validation,
                        "accuracy_validation":accuracy_validation,
                        "accuracy":accuracy,

                        }
        if idx == 0:
            with open("result/randomize_accuracy/weight_comparison.csv", mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=save_dictionary.keys())
                writer.writerow({key: key for key in save_dictionary.keys()})
        with open("result/randomize_accuracy/weight_comparison.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=save_dictionary.keys())
            writer.writerow({key: save_dictionary[key] for key in save_dictionary.keys()})
        print("++++"*100)
    print("\t\n","*"*100)