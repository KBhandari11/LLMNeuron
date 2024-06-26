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

from argparse import Namespace
from utils.dataset import getData
from utils.evaluation import evaluate
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.init as init
from utils.bag_of_words.projection_community import *

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
        low_cpu_mem_usage=True #if args.torch_version >=1.9 else False,
    )
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

def randomize_model(model, modules_list):
    modules_to_reinit = [(int(m.split("_")[0]),m.split("_")[1]) for m in modules_list]
    layers = model.model.layers
    for idx_layer, module in modules_to_reinit:
        layer = layers[idx_layer]
        #print(idx_layer, module)
        for name1, child1 in layer.named_children():
            #print("Name1",name1)
            for name2, child2 in child1.named_children():
                #print("Name2",name2)
                if f"{name1}.{name2}" == f"{module}_proj" or f"{name1}.{name2}" == f"self_{module}_proj" :
                    # Loop over all parameters in the module and apply custom random initialization
                    for param in child2.parameters():
                        if param.requires_grad:  # Ensure the parameter is trainable
                            std = param.std().item()
                            print(std, end=", ")
                            noise = torch.randn_like(param) * std # Small noise
                            param.data += noise
                        ''' if param.dim() > 1:  # Initialize weights
                            init.kaiming_uniform_(param, a=0.01)
                        else:  # Initialize biases
                            init.constant_(param, 0)'''
    print()
    return model

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

def get_high_low_datasets(community, top_skill= 50): 
    return community["dataset"]["all"][:top_skill], community["dataset"]["all"][-top_skill:]

def pick_largest_community(community_data_lists): 
    size = -1
    community_idx = None
    for comm_name, community in community_data_lists.items():
        if len(community["dataset"]) > size:
            size = len(community["dataset"])
            community_idx = comm_name
    non_idx_community = [idx for idx in community_data_lists if idx !=  community_idx]
    return (community_idx, size),community_data_lists[community_idx]["dataset"], flatten_comprehension([community_data_lists[non_idx]["dataset"] for non_idx in non_idx_community])

    



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

if __name__ == "__main__":
    set_random_seed(int(sys.argv[1]))
    sparsity_ratio = "20"
    data = {"iteration":[],"model":[],"pruning_style":[],"community":[],"pruning_ratio":[],"high_20":[],"low_20":[],"high_list":[],"low_list":[]}
    modules_community_dataset,dataset_info_list, dataset_list = get_modulesCommunityDataset(sparsity_ratio)
    #"pruner_style","model","sparsity_ratio","community"
    for idx, model_name in enumerate(modules_community_dataset["model"]):
        print(idx, model_name, modules_community_dataset["pruner_style"][idx])
        community_data_lists = modules_community_dataset["community"][idx]
        for comm_name, community in community_data_lists.items():
            print("Community Name:",comm_name)
            high_module_dataset, low_module_dataset = get_high_low_datasets(community, top_skill=20)
            high_module_dataset, low_module_dataset = get_all_dataset_list(dataset_info_list, high_module_dataset),get_all_dataset_list(dataset_info_list, low_module_dataset)
            model, tokenizer = get_model(model_name)
            reset_model = randomize_model(model, community["modules"])
            #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

            high_module_accuracy = []
            print("HIGH: ",high_module_dataset)
            print("LOW: ",low_module_dataset)
            for dataset_name in high_module_dataset:
                args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=10,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cuda',fine_tune=False,evaluation_size=20)
                _, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                accuracy, _ = evaluate(model=reset_model,tokenizer=tokenizer,testloader=validation_dataset,args=args_dataset)
                high_module_accuracy.append(accuracy[2])
                #high_module_accuracy.append(validation_dataset[0].num_rows)
            low_module_accuracy = []
            for dataset_name in low_module_dataset:
                args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=10,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cuda',fine_tune=False,evaluation_size=20)
                _, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                accuracy, _ = evaluate(model=reset_model,tokenizer=tokenizer,testloader=validation_dataset,args=args_dataset)
                low_module_accuracy.append(accuracy[2])
                #low_module_accuracy.append(validation_dataset[0].num_rows)
            #print("High Module Accuracy",sum(high_module_accuracy)/len(high_module_accuracy))
            print("High Module Accuracy",np.mean(high_module_accuracy),np.std(high_module_accuracy),high_module_accuracy, flush=True)
            #print("Low Module Accuracy",sum(low_module_accuracy)/len(low_module_accuracy))
            print("Low Module Accuracy",np.mean(low_module_accuracy),np.std(low_module_accuracy),low_module_accuracy, flush=True)
            print("-"*20)
            data["iteration"].append(int(sys.argv[1]))
            data["model"].append(model_name)
            data["pruning_style"].append(modules_community_dataset["pruner_style"][idx])
            data["community"].append(comm_name)
            data["pruning_ratio"].append("20")
            data["high_20"].append(sum(high_module_accuracy)/len(high_module_accuracy))
            data["low_20"].append(sum(low_module_accuracy)/len(low_module_accuracy))
            data["high_list"].append(high_module_accuracy)
            data["low_list"].append(low_module_accuracy)
        print("++"*100)
    df = pd.DataFrame(data)
    df.to_csv(f'./result/randomize_accuracy/randomize_data_kaiming_{sys.argv[1]}.csv') 
