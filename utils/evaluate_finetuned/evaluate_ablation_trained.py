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

def is_already_evaluated(results, model_name, pruning_style, community, dataset, finetune, eval_dataset):
    return any(
        r['model'] == model_name and
        r['pruning_style'] == pruning_style and
        r['source_community'] == community and
        r['finetuned_dataset'] == dataset and
        r['finetune_style'] == finetune and
        r['evaluation_dataset'] == eval_dataset
        for r in results
    )

if __name__ == "__main__":
    sparsity_ratio = "20"
    with open("./dataset_info.json", 'r') as openfile:
        dataset_info_list = json.load(openfile)
    existing_runs = pd.read_csv(f'./result/randomize_accuracy/finetuning_result_modified.csv',dtype=str).drop_duplicates(subset=['iteration','model',"pruning_style","community","finetune","dataset"])
    existing_runs = convert_list(existing_runs)
    save_file = "./result/randomize_accuracy/cross_community_evaluation.csv"

    results = []
    if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
        # Load existing data from the CSV into the results list
        saved_df = pd.read_csv(save_file)
        results = saved_df.to_dict('records')
        print("Existing save file found with: ",len(results), " entries.")
        for row_result in results:
            print(f"{row_result['model']}_{row_result['pruning_style']}_{ row_result['finetuned_dataset']}_{row_result['evaluation_dataset']}_{row_result['finetune_style']}")
            print("\t\t",(row_result['test_accuracy'],row_result['test_accuracy_gen']),(row_result['correct'],row_result['correct_gen']),row_result["total"],flush=True)
    all_community = existing_runs['community'].unique().tolist()
    for _, row in existing_runs.iterrows():
        model_name = row['model']
        pruning_style = row['pruning_style']
        community = row['community']
        dataset = row['dataset']
        finetune = row['finetune']
        # Current saved model file
        saved_model = f"{model_name}_{pruning_style}_{community}_{dataset}_{finetune}_5000_5.pt"
        # Evaluate on other community's datasets
        other_community_list = [allcomm  for allcomm in all_community]#1 - community  # Assuming communities are labeled 0 and 1
        print(f"{model_name}_{pruning_style}_{community}_{dataset}_{finetune}")
        for other_community in other_community_list:
            other_datasets = existing_runs[(existing_runs['community'] == other_community) & (existing_runs['model'] == model_name) & (existing_runs['pruning_style'] == pruning_style)]['dataset'].unique()
            for eval_dataset in other_datasets:
                if is_already_evaluated(results, model_name, pruning_style, community, dataset, finetune, eval_dataset):
                    print(f"\tSkipping {eval_dataset}, already evaluated.")
                    continue
                print("\t",eval_dataset)
                eval_dataset_actual = get_all_dataset_list(dataset_info_list, [eval_dataset])[0]
                args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=10,seqlen=1000,model_type="llama",num_process=10,max_length=10,device='cuda',fine_tune=False,evaluation_size=100, seed=0, base_model=model_name,modified_evaluation_dataset=True)
                model,tokenizer = get_model(model_name)
                if other_community != community:
                    if finetune != "Without Finetune":
                        model.load_state_dict(torch.load(f"./checkpoint/{saved_model}"))

                _, test_dataset = getData(tokenizer,dataset_info_list, eval_dataset_actual, args_dataset,modified_evaluation_dataset=True)
                accelerator = Accelerator()
                model, test_dataset = accelerator.prepare(model, test_dataset)
                accuracy, accuracy_gen, predicted_actual,_ = evaluate(model=model,tokenizer=tokenizer,testloader=test_dataset,args=args_dataset)
                print("\t\t",(accuracy[2],accuracy_gen[2]),(accuracy[0],accuracy_gen[0]),accuracy[1],flush=True)
                results.append({
                    'model': model_name,
                    'pruning_style': pruning_style,
                    'finetune_style':finetune,
                    'source_community': community,
                    'target_community': other_community,
                    'finetuned_dataset': dataset, 
                    'evaluation_dataset': eval_dataset,
                    'total': accuracy[1],
                    'correct': accuracy[0],
                    'correct_gen': accuracy_gen[0],
                    'test_accuracy': accuracy[2],
                    'test_accuracy_gen': accuracy_gen[2],
                    'generated':predicted_actual
                })
                save_df = pd.DataFrame(results) 
                save_df.to_csv(save_file, index=False)
                accelerator.free_memory()
                del model
            print("++"*50)
        print("**"*100)
        print("**"*100)
