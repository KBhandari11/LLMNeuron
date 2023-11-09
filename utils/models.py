import torch
import os
from pathlib import Path
from transformers import BertForMultipleChoice 
from transformers import GPT2Tokenizer,  GPT2Model
from transformers import GPTJForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

def GPT(cond =True, model='gpt2'):
    if cond:
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        model =   GPT2Model.from_pretrained(model, pad_token_id=tokenizer.eos_token_id,torch_dtype=torch.float16,  low_cpu_mem_usage=True,device_map="auto")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        model =   GPT2Model.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)
    #model =  GPT2LMHeadModel.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    return model, tokenizer
def GPTJ(model='gptj'):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model =  GPTJForSequenceClassification.from_pretrained(model)
    #AutoTokenizer.pad_token = AutoTokenizer.eos_token
    tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.config.pad_token_id = model.config.eos_token_id
    #model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    '''#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.config.pad_token_id = model.config.eos_token_id
    #model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)'''
    return model, tokenizer
def t5Model(model='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model)
    return model, tokenizer
def BERT(model= "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model)
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = BertForMultipleChoice.from_pretrained(model,torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
    return model, tokenizer
def flatten(nested_list):
    """
    input: nasted_list - this contain any number of nested lists.
    ------------------------
    output: list_of_lists - one list contain all the items.
    """

    list_of_lists = []
    for item in nested_list:
        list_of_lists.extend(item)
    return list_of_lists
## General
def LlamaModel(evalCond,model="meta-llama/Llama-2-7b-hf"):
    print("Model to be used:", model)
    name = flatten([value.split("/") for value in model.split("-")])
    if "13b" in name: 
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    else:
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    if evalCond:
        model = LlamaForCausalLM.from_pretrained(model,torch_dtype=torch.float16,  low_cpu_mem_usage=True)
    else:
        model = LlamaForCausalLM.from_pretrained(model,torch_dtype=torch.float16,  low_cpu_mem_usage=True,device_map="auto")
    #tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.config.pad_token_id = model.config.eos_token_id
    #model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    return model, tokenizer

def LlamaChatModel(evalCond,model="meta-llama/Llama-2-7b-chat-hf"):
    print("CHAT Model to be used:", model)
    #tokenizer = LlamaTokenizer.from_pretrained(model)
    name = flatten([value.split("/") for value in model.split("-")])
    if "13b" in name:
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    else:
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    if evalCond:
        model = LlamaForCausalLM.from_pretrained(model,torch_dtype=torch.float16,  low_cpu_mem_usage=True)
    else:
        model = LlamaForCausalLM.from_pretrained(model,torch_dtype=torch.float16,  low_cpu_mem_usage=True,device_map="auto")
    #tokenizer = LlamaTokenizer.from_pretrained(model"meta-llama/Llama-2-7b-chat-hf")
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.config.pad_token_id = model.config.eos_token_id
    #model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    return model, tokenizer

def getModel(args,evalCond=True):
    if not(os.path.exists(args.save_data+"config.json")):
        #if args.sparsity_ratio != 0.0:
        #    Path(args.save_data).mkdir(parents=True, exist_ok=True)
        model = args.model
    else:
        print("Found a pruned model...")
        model = args.save_data
    if args.model_type == 'gpt':
        return GPT(cond = evalCond,model=model)
    elif args.model_type == 'gptj':
        return GPTJ(model=model)
    elif args.model_type == 't5':
        return t5Model(model=model)
    elif args.model_type == 'bert':
        return BERT(model=model)
    elif args.model_type == 'llama':
       return LlamaModel(evalCond,model=model)
    elif args.model_type == 'llama-chat':
       return LlamaChatModel(evalCond,model=model)
    else:
        raise Exception("the model type is not known. Must be 't5', 'gpt' or 'llama'")