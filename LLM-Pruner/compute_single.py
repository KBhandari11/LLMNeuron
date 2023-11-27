
import os
import gc
import sys
import time
import json
import copy
import random
from argparse import Namespace
from typing import Tuple
import torch.nn as nn 

import csv 


import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts
sys.path.append("../")
from utils.dataset import getData
from utils.evaluation import evaluate

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
def compute_single(logger,dataset_info_list,args): 
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
            model = LlamaForCausalLM.from_pretrained(
                args.base_model,
                low_cpu_mem_usage=True #if args.torch_version >=1.9 else False
            )
            if args.device != "cpu":
                model.half()
            model.to(args.device)
            pruner_type = args.pruner_type.lower()
            assert pruner_type in ['random', 'l2', 'l1', 'taylor']

            for param in model.parameters():
                param.requires_grad_(True)
            before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            forward_prompts = torch.tensor([
                [    1,   306,  4658,   278,  6593,   310,  2834,   338],
                [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
            ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

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
                kwargs = {
                    "importance": imp,
                    "global_pruning": args.global_pruning,
                    "iterative_steps": args.iterative_steps,
                    "ch_sparsity": args.pruning_ratio, 
                    "ignored_layers":[],
                    "channel_groups": {
                    },
                    "consecutive_groups": {
                        layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
                    },
                    "customized_pruners": {
                        LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                    },
                    "root_module_types": None, 
                    "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                                    [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
                }
                logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
                logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

                pruner = tp.pruner.MetaPruner(
                    model,
                    forward_prompts,
                    **kwargs
                )
                model.zero_grad()
                logger.log("Start Pruning")
                for i in range(args.iterative_steps):
                    if pruner_type in ['taylor']:
                        args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=400,model_type="llama",num_process=10,max_length=0,device='cpu',fine_tune=False)
                        example_prompts, _ = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                        #example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
                        logger.log("Start Backwarding in iterative steps = {}...".format(i))
                        if args.taylor in ['param_mix', 'param_second']:
                            for example_train, example_label in example_prompts:
                                batch_input = example_train.unsqueeze(0).to(args.device) 
                                batch_label = example_label.unsqueeze(0).to(args.device) 
                                loss = model(batch_input, labels=batch_label).loss
                                loss.backward()
                                del(batch_input)
                                del(batch_label)
                                for module_param in model.parameters():
                                    module_param.grad = module_param.grad * module_param.grad / args.num_examples
                                    if hasattr(module_param, 'acc_grad'):
                                        module_param.acc_grad += module_param.grad
                                    else:
                                        module_param.acc_grad = copy.deepcopy(module_param.grad)
                                model.zero_grad()
                                del loss.grad
                        loss = 0
                        c = 0 
                        for batch_input ,batch_label in example_prompts:
                            c+=1
                            input = batch_input.to(args.device)
                            label = batch_label.to(args.device) 
                            loss += model(input, labels=label).loss
                            del(input)
                            del(label)
                        loss.backward()
                        loss = loss/len(example_prompts)
                        logger.log("Loss = {}".format(loss))
                    pruner.step()
                    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
                
                    # modify inferece-related attributes
                    for layer in model.model.layers:
                        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

                # Clean the gradient in the model
                model.zero_grad()
                for name, module in model.named_parameters():
                    if 'weight' in name:
                        module.grad = None

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
                        loss = 0
                        for train_data ,label_data in example_prompts:
                            input = train_data.to(args.device)
                            label = label_data.to(args.device)
                            loss += model(input, labels=label).loss
                            del(input)
                            del(label)
                        loss = loss /len(example_prompts)
                        logger.log("Loss = {}".format(loss))
                        loss.backward()
                
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
            
            if args.save_distribution:
                if isinstance(dataset_name,list):
                    sparsity, all_distribution["|W|_F"][dataset_name[-1].split('/')[-1]] , all_distribution["|W|_0"][dataset_name[-1].split('/')[-1]]  = create_distribution_llm_pruner(model)
                    logger.log(f"{dataset_name[-1].split('/')[-1]}: Total Sparsity {sparsity}")
                else:
                    sparsity, all_distribution["|W|_F"][dataset_name.split('/')[-1]] , all_distribution["|W|_0"][dataset_name.split('/')[-1]]  = create_distribution_llm_pruner(model)
                    logger.log(f"{dataset_name.split('/')[-1]}: Total Sparsity {sparsity}")

                torch.cuda.empty_cache()

            if args.test_after_train:
                if args.eval_device != "cpu":
                    model.half()
                model.to(args.eval_device)

                model.config.pad_token_id = tokenizer.pad_token_id = 0 
                model.config.bos_token_id = 1
                model.config.eos_token_id = 2
                logger.log("\n==================Generation Results After Pruning================\n")
                
                args_valid = Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=400,model_type="llama",num_process=10,max_length=0,device='cuda',fine_tune=False)
                model.eval()
                print(dataset_name)
                logger.log(dataset_name)
                torch.cuda.empty_cache()
                _, valid_dataloader = getData(tokenizer,dataset_info_list, dataset_name, args_valid)
                ppl, saveResult = evaluate(model,tokenizer, valid_dataloader, args_valid)
                logger.log(f"Accuracy on {dataset_name}: {ppl}")
                logger.log("*"*100)
                fname= args.save_ckpt_log_name+ "/"+ dataset_name.split('/')[-1]+".csv"
                with open(fname, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(saveResult)
            logger.log("\n==================Finish================\n")
            logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
    if args.save_distribution:
        return all_distribution