
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
from torch.utils.data import TensorDataset,DataLoader
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM

from transformers.models.llama.modeling_llama import LlamaRMSNorm
import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as Pruner

sys.path.append("../")
from utils.dataset import getData
from utils.evaluation import evaluate

'''def collate_fn(batch):
  return_data = {
      'input_ids': torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
      'labels': torch.stack([torch.tensor(x["labels"]) for x in batch])
    }
  return return_data'''
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
def free_mem(model):
    del_grad(model)
    del model
    torch.cuda.empty_cache()

def convert_dataloader(train_dataloader_list):
    tensor_data = torch.stack([x.squeeze(0) for x,_,_ in train_dataloader_list],dim=0) # transform to torch tensor
    tensor_atten = torch.stack([y.squeeze(0) for _,y,_ in train_dataloader_list],dim=0) # transform to torch tensor
    tensor_label = torch.stack([z.squeeze(0) for _,_,z in train_dataloader_list],dim=0)
    #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = TensorDataset(tensor_data,tensor_atten,tensor_label) # create your datset
    #train_dataset = DataDataset(train_dataloader_list)
    train_kwargs = {'batch_size': len(train_dataloader_list), "drop_last":False}
    cuda_kwargs = {'shuffle': True}
    train_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset,**train_kwargs)
    return train_loader

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


    
def prune_llama3(logger,dataset_info_list,args): 
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
        model = AutoModelForCausalLM.from_pretrained( args.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
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
            kwargs = {
                "importance": imp,
                "global_pruning": args.global_pruning,
                "iterative_steps": args.iterative_steps,
                "ch_sparsity": args.pruning_ratio, 
                "ignored_layers":[],
                "channel_groups": {
                },
                "consecutive_groups": {
                    layer.self_attn.k_proj: layer.self_attn.head_dim for layer in model.model.layers #adding module for dataparallel
                },
                "customized_pruners": {
                    LlamaRMSNorm: Pruner.hf_rmsnorm_pruner,
                },
                "root_module_types": None, 
                "root_instances": [model.model.layers[i].self_attn.k_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
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
                    args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cuda',fine_tune=False,evaluation_size=50, seed=0)
                    example_prompts, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                    logger.log("Start Backwarding in iterative steps = {}...".format(i))
                    if args.taylor in ['param_mix', 'param_second']:
                        loss_value = 0
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
                    loss_value = 0
                    train_loader = convert_dataloader(example_prompts)
                    for input in example_prompts:
                        data,atten, target = input[0].to(args.device), input[2].to(args.device), input[2].to(args.device)
                        loss = model(input_ids=data,attention_mask=atten,labels=target).loss
                        loss.backward()
                        loss_value += loss.item()
                        del(input)
                    loss = loss_value /len(example_prompts)
                    logger.log("Loss = {}".format(loss))
                    '''for idx, input in enumerate(train_loader):
                        data,atten, target = input[0].to(args.device), input[2].to(args.device), input[2].to(args.device)
                        loss = model(input_ids=data,attention_mask=atten,labels=target).loss
                        loss.backward()'''

                else:
                    args_dataset =  Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cuda',fine_tune=False,evaluation_size=50, seed=0)
                    example_prompts, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                if pruner_type in ['taylor']:
                    del loss
                    del loss_value
                    del train_loader 
                logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024)) 
                pruner.step()
                after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
            
                # modify inferece-related attributes
                for layer in model.model.layers:
                    layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
                    layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim
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
                    LlamaRMSNorm: Pruner.hf_rmsnorm_pruner,
                    #LlamaAttention: Pruner.hf_attention_pruner,
                },
                "root_module_types": [LlamaRMSNorm],
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
                    args_dataset =  Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cuda',fine_tune=False,evaluation_size=50, seed=0)
                    example_prompts, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                    #example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
                    logger.log("Start Backwarding in iterative steps = {}...".format(i))
                    loss_value = 0
                    '''train_loader = convert_dataloader(example_prompts)
                    for idx, input in enumerate(train_loader):
                        loss = model(input_ids=input["input_ids"].to(args.device),labels=input["labels"].to(args.device)).loss
                        loss.backward()
                        del(input)
                        logger.log("Loss = {}".format(loss.item()))'''
                    for input in example_prompts:
                        data,atten, target = input[0].to(args.device), input[2].to(args.device), input[2].to(args.device)
                        loss = model(input_ids=data,attention_mask=atten,labels=target).loss
                        loss.backward()
                        loss_value += loss.item()
                        del(input)
                    loss = loss_value /len(example_prompts)
                    logger.log("Loss = {}".format(loss))
                else:
                    args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=args.num_examples,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cuda',fine_tune=False,evaluation_size=50, seed=0)
                    example_prompts, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
                if pruner_type in ['taylor']:
                    del loss
                    del loss_value
                logger.log("Pruner Step Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
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
            #del train_loader
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
            model.config.pad_token_id = tokenizer.pad_token_id = 0 
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
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
        free_mem(model)
        torch.cuda.empty_cache()
    if args.save_distribution:
        return all_distribution