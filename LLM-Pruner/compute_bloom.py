
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

from transformers import AutoTokenizer
from LLMPruner.models.hf_bloom.modeling_bloom import BloomForCausalLM
from LLMPruner.models.helper import reorder_qkv


import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as Pruner

sys.path.append("../")
from utils.dataset import getData
from utils.evaluation import evaluate

def collate_fn(batch):
  return_data = {
      'input_ids': torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
      'labels': torch.stack([torch.tensor(x["labels"]) for x in batch])
    }
  return return_data


def convert_dataloader(lists):
    my_dict = {}
    my_dict["input_ids"]= [x[0] for x,_ in lists]
    my_dict["labels"]= [y[0] for _,y in lists]
    dataset = Dataset.from_dict(my_dict)
    kwargs = {'batch_size': len(lists),'collate_fn':collate_fn,'shuffle': True}
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
    layers = model.transformer.h
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
            layer_values_2.append(torch.linalg.matrix_norm(W, ord=float("Inf")).item()) #|W|_inf norm
            layer_values_0.append((W.numel() - (W==0).sum().item())) #|W|_0 norm
        distribution_F.append(layer_values_F)
        distribution_2.append(layer_values_2)
        distribution_0.append(layer_values_0)
    return  np.array(distribution_F),np.array(distribution_2),np.array(distribution_0),float(count)/total_params

def get_dataset_list(dataset_list):
    dataname = []
    for data in dataset_list:
        if "subset" not in dataset_list[data].keys():
            dataname.append(data)
        else:
            for subset in dataset_list[data]["subset"]:
                dataname.append([data,subset])
    return dataname


    
def prune_bloom(logger,dataset_info_list,args): 
    all_distribution = {}
    dataset_list = get_dataset_list(dataset_info_list)
    for dataset_name in dataset_list:
            distribution = {"|W|_F":None,"|W|_inf":None,"|W|_0":None,"Accuracy":None}
            logger.log("\n"+"*"*100+"*"*100+"\n")
            if isinstance(dataset_name,list):
                logger.log("DATASET: "+ " ".join(dataset_name))
            else:
                logger.log("DATASET: "+dataset_name)
            torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
            model = BloomForCausalLM.from_pretrained(
                args.base_model,
                low_cpu_mem_usage=True #if args.torch_version >=1.9 else False,
            )
            for layer in model.transformer.h:
                att_module = layer.self_attention
                reorder_qkv(att_module.query_key_value, att_module.head_dim, att_module.num_heads)
        
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
            #.to("cpu")
            if pruner_type == 'random':
                imp = tp.importance.RandomImportance()
            elif pruner_type == 'l1':
                imp = Pruner.MagnitudeImportance(p=1)
            elif pruner_type == 'l2':
                imp = Pruner.MagnitudeImportance(p=2)
            elif pruner_type == 'taylor':
                imp = Pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
            else:
                raise NotImplementedError
            logger.log("Use {} pruner...".format(pruner_type))
            if args.block_wise:
                def forward_fn(model, example_inputs):
                    return model(example_inputs)

                kwargs = {
                    "importance": imp,
                    "global_pruning": args.global_pruning,
                    "iterative_steps": args.iterative_steps,
                    "ch_sparsity": args.pruning_ratio, 
                    "ch_sparsity_dict": {
                        model.transformer.h[i].self_attention.query_key_value: args.pruning_ratio / 3 for i in range(args.block_attention_layer_start, args.block_attention_layer_end)
                    },
                    "forward_fn": forward_fn,
                    "ignored_layers":[],
                    "channel_groups": {
                        #layer.self_attn.W_pack: 3 for layer in model.model.layers
                    },
                    "consecutive_groups": {
                        layer.self_attention.query_key_value: layer.self_attention.head_dim for layer in model.transformer.h
                    },
                    "customized_pruners": {},
                    "root_module_types": None, 
                    "root_instances": [model.transformer.h[i].mlp.dense_h_to_4h for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)] +
                                    [model.transformer.h[i].self_attention.query_key_value for i in range(args.block_attention_layer_start, args.block_attention_layer_end)],
                    "enable_index_mapping": True
                                    
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
                        args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cpu',fine_tune=False)
                        example_prompts, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                        logger.log("Start Backwarding in iterative steps = {}...".format(i))
                        if args.taylor in ['param_mix', 'param_second']:
                            for example_train, example_label in example_prompts:
                                batch_input = example_train.unsqueeze(0).to(args.device) 
                                batch_label = example_label.unsqueeze(0).to(args.device) 
                                loss = model(batch_input, labels=batch_label).loss
                                logger.log("Loss = {}".format(loss))
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
                        train_loader = convert_dataloader(example_prompts)
                        '''for batch_input ,batch_label in example_prompts:
                            input = batch_input.to(args.device)
                            label = batch_label.to(args.device)
                            loss += model(input, labels=label ).loss
                            del(input)
                            del(label)
                        loss.backward()
                        loss = loss/len(example_prompts)'''
                        for idx, input in enumerate(train_loader):
                            loss = model(input_ids=input["input_ids"].to(args.device),labels=input["labels"].to(args.device)).loss
                            loss.backward()
                            logger.log("Loss = {}".format(loss))

                    for group in pruner.step(interactive=True):
                        group.prune()
                    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
                
                    # modify inferece-related attributes
                    for layer in model.transformer.h:
                        layer.self_attention.hidden_size = layer.self_attention.query_key_value.weight.shape[0] // 3
            
                # Clean the gradient in the model
                model.zero_grad()
                for name, module in model.named_parameters():
                    if 'weight' in name:
                        module.grad = None
                del train_loader
                del pruner
                del loss
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
                if args.block_wise:
                    print("Saving Block Wise")
                    pruneType = 'block'
                    torch.save(model.state_dict(), f"/data/Kushal/MLNeuron/checkpoints/llama-7b/LLM_Pruner/block/{int(args.pruning_ratio*100)}/model_{dataset_name[-1].split('/')[-1]}.pt")
                elif args.layer_wise:
                    print("Saving Layer Wise")
                    pruneType = 'layer'
                    torch.save(model.state_dict(), f"/data/Kushal/MLNeuron/checkpoints/llama-7b/LLM_Pruner/layer/{int(args.pruning_ratio*100)}/model_{dataset_name[-1].split('/')[-1]}.pt")
        
            if args.save_distribution:
                if isinstance(dataset_name,list):
                    distribution["|W|_F"] ,distribution["|W|_inf"], distribution["|W|_0"],sparsity  = create_distribution_llm_pruner(model)
                    logger.log(f"{dataset_name[-1].split('/')[-1]}: Total Sparsity {sparsity}")
                    accuracy, _ = evaluate(model=model,tokenizer=tokenizer,testloader=validation_dataset,args=args)
                    logger.log(f"{dataset_name[-1].split('/')[-1]}: Total Accuracy {accuracy}")
                    distribution["Accuracy"] = accuracy
                    all_distribution[dataset_name[-1].split('/')[-1]]= distribution
                else:
                    distribution["|W|_F"],distribution["|W|_inf"], distribution["|W|_0"],sparsity = create_distribution_llm_pruner(model)
                    logger.log(f"{dataset_name.split('/')[-1]}: Total Sparsity {sparsity}")
                    accuracy,_ = evaluate(model=model,tokenizer=tokenizer,testloader=validation_dataset,args=args)
                    logger.log(f"{dataset_name.split('/')[-1]}: Accuracy {accuracy}")
                    distribution["Accuracy"] = accuracy
                    all_distribution[dataset_name.split('/')[-1]] = distribution
                torch.cuda.empty_cache()
            logger.log("\n==================Finish================\n")
            logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
    torch.cuda.empty_cache()
    del model
    if args.save_distribution:
        return all_distribution