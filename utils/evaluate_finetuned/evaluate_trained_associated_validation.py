
'''
Add only new made changed value:

'''
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

def compute_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_norm += torch.norm(param).item()
    return l2_norm

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
    return existing_runs

def check_existing_runs(existing_runs,  model_name, comm_name, pruner_style, pruning_ratio, dataset, finetune):
    # Filter the DataFrame for rows that match the specified values
    existing_runs=existing_runs.drop_duplicates(subset=['iteration','model',"pruning_style","pruning_ratio","community","finetune","dataset"])
    condition = (
        (existing_runs['model'] == model_name) &
        (existing_runs['community'] == str(comm_name)) &
        (existing_runs['pruning_style'] == pruner_style) &
        (existing_runs['pruning_ratio'] == str(pruning_ratio)) &
        (existing_runs['dataset'] == dataset) &
        (existing_runs['finetune'] == finetune)
    )
    exists = not existing_runs[condition].empty
    
    # Check if any row satisfies the condition
    return exists

if __name__ == "__main__":
    with open("./dataset_info.json", 'r') as openfile:
        dataset_info_list = json.load(openfile)
    existing_runs = pd.read_csv(f'./result/randomize_accuracy/finetuning_result_modified.csv',dtype=str).drop_duplicates(subset=['iteration','model',"pruning_style","community","finetune","dataset"])
    existing_runs = existing_runs.sort_values(by=['iteration','model',"pruning_style","community","dataset","finetune"])
    existing_runs = convert_list(existing_runs)
    save_file = "/gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/result/randomize_accuracy/finetuning_result_reevaluate.csv"
    results = []
    if os.path.exists(save_file):
        # Load existing data from the CSV into the results list
        saved_df = pd.read_csv(save_file,dtype=str)
        results = saved_df.to_dict('records')
        print("Existing save file found with: ",len(results), " entries.")
        for row_result in results:
            print(f"{row_result['model']}_{row_result['pruning_style']}_{ row_result['community']}_{row_result['dataset']}_{row_result['finetune']}")
            print("\t\t",(row_result['test_accuracy'],row_result['test_accuracy_gen']),(row_result['correct'],row_result['correct_gen']),row_result["total"],row_result["final_model_l2"],flush=True)
    else:
        saved_df = pd.DataFrame(results) 
    existing_runs_dict = existing_runs.to_dict(orient='records')
    for row in existing_runs_dict:
        model_name = row['model']
        pruning_style = row['pruning_style']
        community = row['community']
        eval_dataset = row['dataset']
        pruning_ratio = row['pruning_ratio']
        finetune = row['finetune']
        if check_existing_runs(saved_df,  model_name, community, pruning_style, pruning_ratio, eval_dataset, finetune):
            continue
        # Current saved model file
        saved_model = f"{model_name}_{pruning_style}_{community}_{eval_dataset}_{finetune}_5000_5.pt"
        # Evaluate on other community's datasets
        print(f"{model_name}_{pruning_style}_{community}_{eval_dataset}_{finetune}")
        
        eval_dataset_actual = get_all_dataset_list(dataset_info_list, [eval_dataset])[0]
        args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=10,seqlen=1000,model_type="llama",num_process=10,max_length=10,device='cuda',fine_tune=False,evaluation_size=100, seed=0, base_model=model_name)
        model,tokenizer = get_model(model_name)
        if os.path.isfile(f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/checkpoint/{saved_model}"):
            print("\tFile Exists...")
            model.load_state_dict(torch.load(f"/gpfs/u/home/LLMG/LLMGbhnd/scratch/checkpoint/{saved_model}"))

        _, test_dataset = getData(tokenizer,dataset_info_list, eval_dataset_actual, args_dataset,modified_evaluation_dataset=True)
        accelerator = Accelerator()
        model, test_dataset = accelerator.prepare(model, test_dataset)
        accuracy, accuracy_gen, predicted_actual,_ = evaluate(model=model,tokenizer=tokenizer,testloader=test_dataset,args=args_dataset)
        
        row["final_model_l2"] = compute_l2_norm(model)
        print("\t\t",(accuracy[2],accuracy_gen[2]),(accuracy[0],accuracy_gen[0]),accuracy[1],row["final_model_l2"],flush=True)

        row['total'] = accuracy[1]
        row['correct'] = accuracy[0]
        row['correct_gen'] = accuracy_gen[0]
        row['generated'] =predicted_actual
        row['test_accuracy'] = accuracy[2]
        row['test_accuracy_gen'] = accuracy_gen[2]

        results.append(row)
        save_df = pd.DataFrame(results) 
        save_df.to_csv(save_file, index=False)
        accelerator.free_memory()
        del model
        print("++"*50)
