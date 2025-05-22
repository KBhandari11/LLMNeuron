import os
import sys
import random
import argparse
import pandas as pd

import csv 
import json
import re 

import ast
import torch
import numpy as np
import torch.nn as nn 
from collections import OrderedDict

import functools
from argparse import Namespace
from utils.dataset import getData
from utils.evaluation import evaluate, mcq_token_index,computeLogits
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.optim import AdamW,Adadelta
from utils.bag_of_words.projection_community import create_projection_network
from accelerate import Accelerator

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

def compute_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_norm += torch.norm(param).item()
    return l2_norm

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

def convert_list(existing_runs):
    existing_runs['modules'] = existing_runs['modules'].apply(ast.literal_eval)
    existing_runs['loss_training'] = existing_runs['loss_training'].apply(ast.literal_eval)
    existing_runs['loss_validation'] = existing_runs['loss_validation'].apply(ast.literal_eval)
    existing_runs['community'] = existing_runs['community'].apply(int)
    existing_runs = existing_runs.sort_values(['model', 'pruning_style','community', 'finetune'])
    return existing_runs

def is_already_evaluated(results, model_name, pruning_style, community, dataset, finetune):
    return any(
        r['model'] == model_name and
        r['pruning_style'] == pruning_style and
        r['community'] == community and
        r['dataset'] == dataset and
        r['finetune'] == finetune
        for r in results
    )
def save_result_csv(filename, data):
    data_keys = ["iteration","model","pruning_style","community","pruning_ratio","dataset","finetune","modules_size","modules","rank","training_dataset_size","validation_dataset_size","loss_training","model_l2_0","loss_validation","accuracy_validation","model_l2","total","correct","accuracy_test","correct_gen","accuracy_test_gen","generated_answer"]
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_keys)
        writer.writerow({key: data[key] for key in data_keys})#this is redundant just to keep order consistent

if __name__ == "__main__":
    sparsity_ratio = "20"
    with open("./dataset_info.json", 'r') as openfile:
        dataset_info_list = json.load(openfile)
    existing_runs = pd.read_csv(f'./result/randomize_accuracy/randomize_data_new_kl_longer_2_old.csv',dtype=str).drop_duplicates(subset=['iteration','model',"community","dataset"])
    existing_runs = convert_list(existing_runs)
    save_file = "./result/randomize_accuracy/no_module_accuracy.csv"

    results = []
    if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
        # Load existing data from the CSV into the results list
        saved_df = pd.read_csv(save_file)
        results = saved_df.to_dict('records')
        print("Existing save file found with: ",len(results), " entries.")
    
    for save_dictionary in existing_runs.to_dict(orient="records"):
        model_name = save_dictionary['model']
        pruning_style = save_dictionary['pruning_style']
        community = save_dictionary['community']
        dataset = save_dictionary['dataset']
        save_dictionary['finetune'] = "No Modules"
        finetune=save_dictionary['finetune'] 
        # Current saved model file
        # Evaluate on other community's datasets
        if is_already_evaluated(results, model_name, pruning_style, community, dataset, finetune):
                    print(f"\tSkipping {dataset}, already evaluated.")
                    continue
        print(f"{model_name}_{pruning_style}_{community}_{dataset}_{finetune}")

        eval_dataset_actual = get_all_dataset_list(dataset_info_list, [dataset])[0]
        args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=10,seqlen=2000,model_type="llama",num_process=10,max_length=10,device='cuda',fine_tune=False,evaluation_size=100, seed=0, base_model=model_name)
        model,tokenizer = get_model(model_name)
        _, test_dataset = getData(tokenizer,dataset_info_list, eval_dataset_actual, args_dataset)
        accelerator = Accelerator()
        model, test_dataset = accelerator.prepare(model, test_dataset)
        accuracy,accuracy_gen,predicted_actual,_ = evaluate(model=model,tokenizer=tokenizer,testloader=test_dataset,args=args_dataset)
        print((accuracy[2],accuracy_gen[2]),(accuracy[0],accuracy_gen[0]),accuracy[1],flush=True)
        
        save_dictionary["loss"] = None 
        save_dictionary["magnitude_list"] = None
        save_dictionary["val_loss_list"] = None
        save_dictionary["val_accuracy_list"] = None
        save_dictionary["final_model_l2"] = compute_l2_norm(model)

        save_dictionary['total'] = accuracy[1]
        save_dictionary['correct'] = accuracy[0]
        save_dictionary['accuracy_test'] = accuracy[2]
        save_dictionary['correct_gen'] = accuracy_gen[0]
        save_dictionary['accuracy_test_gen'] = accuracy_gen[2]
        save_dictionary['generated_answer'] =predicted_actual
        save_result_csv(save_file,save_dictionary)

        accelerator.free_memory()
        del model
        print("++"*100)
