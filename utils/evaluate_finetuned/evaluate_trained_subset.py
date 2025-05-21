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

from argparse import Namespace
from utils.dataset import getData
from utils.evaluation import evaluate
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.init as init
from utils.bag_of_words.projection_community import *

from pynvml import *
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator
from transformers import AdamW

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
        low_cpu_mem_usage=True, 
        device_map="auto"
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

def train_model(train_dataloader_list,model,num_epochs = 5):
    if len(train_dataloader_list) == 0:
        print("No data within dataset")
        print("No data within dataset",file=sys.stderr)
        return model 
    optimizer = AdamW(model.parameters(), lr=3e-5)
    model.train()
    #model.to('cuda')
    accelerator = Accelerator()
    tensor_data = torch.stack([x.squeeze(0) for x,_ in train_dataloader_list],dim=0) # transform to torch tensor
    tensor_label = torch.stack([y.squeeze(0) for _,y in train_dataloader_list],dim=0)
    train_dataloader = TensorDataset(tensor_data,tensor_label) # create your datset
    train_dataloader = DataLoader(train_dataloader) # create your dataloader
    train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                #loss = model(batch_input, labels=batch_label).loss
                outputs = model(batch[0], labels=batch[1])
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f'| WARNING: ran out of memory, retrying batch without that example {batch_idx}',file=sys.stderr)
                    del_grad(model)
                    torch.cuda.empty_cache()
                    if 'loss' in vars() or 'loss' in globals():
                        del loss
                        del accelerator
                        del train_dataloader
                        del optimizer
                    train_dataloader_list.pop(batch_idx)
                    return train_model(train_dataloader_list,model,num_epochs)
                else:
                    raise e
    del loss
    del accelerator
    del train_dataloader
    del optimizer
    return model


def freeze_and_train_model(model, modules_list,train_dataloader):
    model_new =  freeze_model(model, modules_list)
    model_train  = train_model(train_dataloader,model_new,num_epochs = 5)
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
    data = {"iteration":[],"model":[],"pruning_style":[],"community":[],"pruning_ratio":[],"dataset":[],"accuracy":[],"rank_kl":[],"rank_network":[],"modules":[]}
    modules_community_dataset,dataset_info_list, dataset_list = get_modulesCommunityDataset(sparsity_ratio)
    #"pruner_style","model","sparsity_ratio","community"
    for idx, model_name in enumerate(modules_community_dataset["model"]):
        print(idx, model_name, modules_community_dataset["pruner_style"][idx])
        community_data_lists = modules_community_dataset["community"][idx]
        community_data_lists["-1"] = None
        #community_data_lists["-2"] = None
        community_data_lists = OrderedDict(sorted(community_data_lists.items(), key=lambda t: int(t[0])))
        module_dataset = random.sample(dataset_list, 10)
        module_dataset_info_format = get_all_dataset_list(dataset_info_list, module_dataset)
        print("Random Dataset",module_dataset, flush=True)
        for comm_name, community in community_data_lists.items():
            print("Community Name:",comm_name)
            #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            module_accuracy = []
            rank_list = []
            for dataset_name_label,dataset_name in zip(module_dataset, module_dataset_info_format):
                model, tokenizer = get_model(model_name)
                args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=100,seqlen=500,model_type="llama",num_process=10,max_length=100,device='cuda',fine_tune=False,evaluation_size=20, seed=0)
                train_loader, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                if comm_name == "-1":
                    finetuned_model = model#train_model(train_loader,model,num_epochs = 3)
                    module_list = []
                    rank_kl = None
                    rank_network = None
                elif comm_name == "-2":
                    finetuned_model = train_model(train_loader,model,num_epochs = 5)
                    module_list = []
                    rank_kl = None
                    rank_network = None
                else:
                    module_list =  community["modules"]
                    finetuned_model= freeze_and_train_model(model, module_list, train_loader)
                    rank_kl = modules_community_dataset["community"]["kl"][idx][comm_name]["dataset"]["all"].index(dataset_name_label)
                    rank_network = modules_community_dataset["community"]["network"][idx][comm_name]["dataset"].index(dataset_name_label)
                print((dataset_name_label,rank_kl,rank_network),end=": ")
                accuracy, _ = evaluate(model=finetuned_model,tokenizer=tokenizer,testloader=validation_dataset,args=args_dataset)
                print(accuracy,end=", ")

                module_accuracy.append(accuracy)
                rank_list.append([rank_kl,rank_network])
                free_mem(finetuned_model)
                free_mem(model)
                del finetuned_model
                del model
                

                data["iteration"].append(int(sys.argv[1]))
                data["model"].append(model_name)
                data["pruning_style"].append(modules_community_dataset["pruner_style"][idx])
                data["community"].append(comm_name)
                data["pruning_ratio"].append(sparsity_ratio)
                data["dataset"].append(dataset_name_label)
                data["accuracy"].append(accuracy[2])
                data["rank_kl"].append(rank_kl)
                data["rank_network"].append(rank_network)
                data["modules"].append(module_list)
            print()
            print(module_list)
            print("Module Accuracy",comm_name,module_accuracy, flush=True)
            print("Module Rank",comm_name,rank_list, flush=True)
        print("++"*100)
        df = pd.DataFrame(data)
        df.to_csv(f'./result/randomize_accuracy/randomize_data_new_{sys.argv[1]}.csv') 

    df = pd.DataFrame(data)
    df.to_csv(f'./result/randomize_accuracy/randomize_data_new_{sys.argv[1]}.csv') 
