import re
import random
import torch
from datasets import load_dataset 
from .tokenizer import getDataLoader

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def getOptions(options_string):
    pattern =  r"([a-e])\s*\)\s*([\w\s/]+)"
    parse = re.findall(pattern, options_string)
    values = []
    for u, v in parse:
        values.append(v)
    '''if values == []:
        for prefix in prefixes:
            match = re.search(f"{re.escape(prefix)}\s*(\d+)",options_string)
            if match:
                value = match.group(1)
                values.append(value)'''
    return values

def adjustBERT(examples):
    if datasetName == "math_qa":
        option = getOptions(examples[key[1]])
        examples["label"] = ord(examples[key[2]].upper())-ord('A')
    elif datasetName == "commonsense_qa":
        option = examples[key[1]]["text"]
        examples["label"] = ord(examples[key[2]].upper())-ord('A')
    elif datasetName == "derek-thomas/ScienceQA":
        option = examples[key[1]]
        examples["label"] = examples[key[2]]
    examples["input"] = [[examples[key[0]], str(opt)] for opt in option]
    return examples
def insertOption(option):
    return [f"{chr(i+ord('A'))}. {opt}" for i,opt in enumerate(option)]
def adjustAutoRegressive(examples):
    if datasetName == "math_qa":
        option = getOptions(examples[key[1]])
        examples["label"] = ord(examples[key[2]].upper())-ord('A')
    elif datasetName == "commonsense_qa":
        option = examples[key[1]]["text"]
        examples["label"] = ord(examples[key[2]].upper())-ord('A')
    elif datasetName == "derek-thomas/ScienceQA":
        option = examples[key[1]]
        examples["label"] = examples[key[2]]
    elif datasetName == "EleutherAI/truthful_qa_mc":
        option = examples[key[1]]
        examples["label"] = examples[key[2]]
    elif datasetName == "tasksource/bigbench":
        option = examples[key[1]]
        examples["label"] = examples[key[2]].index(1)
    elif datasetName == "cais/mmlu":
        option = examples[key[1]]
        examples["label"] = examples[key[2]]
    option = insertOption(option)
    examples["target"] = chr(int(examples["label"])+ord('A')) #option[int(examples["label"])] #chr(int(examples["label"])+ord('A'))#
    examples["input"] = fewshotPrompt + f"Question: {examples[key[0]]} \n " + (" \n ").join(option)+" Answer: "
    examples["input_no_few"] = f"Question: {examples[key[0]]} \n " + (" \n ").join(option)+" Answer: "
    return examples

def modifyDataset(dataset,keys,fewshot, prefix, dataset_name, args):
    global key, prefixes,datasetName, fewshotPrompt
    key = keys
    prefixes = prefix
    datasetName = dataset_name
    fewshotPrompt = fewshot
    if args.model_type == 'bert':
        dataset = dataset.map(adjustBERT,num_proc=args.num_process)
    else:
        dataset = dataset.map(adjustAutoRegressive,num_proc=args.num_process)

    return dataset

def get_data(dataset_name,dataset_list, tokenizer, args, seed=0):
    # Load train and validation datasets
    print("*"*30)
    print("Loading Dataset")
    if dataset_name == "cais/mmlu":
        traindata = load_dataset(dataset_name,'all', split="auxiliary_train") 
        valdata = load_dataset(dataset_name, 'all',split="validation") 
    elif dataset_name == "tasksource/bigbench":
        traindata = load_dataset(dataset_name,'abstract_narrative_understanding', split="train") 
        valdata = load_dataset(dataset_name, 'abstract_narrative_understanding',split="validation") 
    elif dataset_name == "EleutherAI/truthful_qa_mc":
        traindata = load_dataset(dataset_name, split="validation") 
        valdata = load_dataset(dataset_name, split="validation") 
    else:
        traindata = load_dataset(dataset_name, split="train") 
        valdata = load_dataset(dataset_name, split="validation") 
    #traindata.cleanup_cache_files()
    #valdata.cleanup_cache_files()
    traindata = modifyDataset(traindata,dataset_list[dataset_name]["keys"], "",dataset_list[dataset_name]["prefixes"],dataset_name,args)
    valdata = modifyDataset(valdata,dataset_list[dataset_name]["keys"],dataset_list[dataset_name]["fewshot_prompt"],dataset_list[dataset_name]["prefixes"],dataset_name,args)
    print("*"*30)
    print("Generating Samples")
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    if args.do_train_both:
        num_dataset = int(args.nsamples /2)
    else:
        num_dataset = args.nsamples
    for num in range(num_dataset):
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata[i]['input'], return_tensors='pt')
        label = tokenizer(traindata[i]['target'], return_tensors='pt')
        i = trainenc.input_ids.shape[1]#random.randint(0, trainenc.input_ids.shape[1] - args.seqlen - 1)
        #trainenc.input_ids = torch.cat((trainenc.input_ids, tar.input_ids), 1)
        inp = trainenc.input_ids 
        tar = inp.clone()
        tar = torch.cat((tar, label.input_ids), 1)
        #tar[:, :i] = -100
        inp = torch.nn.functional.pad(inp, (0, args.seqlen - inp.size(1)))
        tar = torch.nn.functional.pad(tar, (0, args.seqlen - tar.size(1)))
        trainloader.append((inp, tar))
    print("*"*30)
    print("Prepare Validation Dataset")
    valloader = getDataLoader(tokenizerGiven=tokenizer, dataset=valdata, args=args)
    return trainloader, (valdata,valloader)

def getData(tokenizer,dataset_list, dataset_name, args):
    if args.do_train_both:
        train_dataset = []
        valid_dataset = []
        for dataset in dataset_name:
            train, _ = get_data(dataset,dataset_list,tokenizer, args)
            train_dataset += train
    else:
        train_dataset, valid_dataset = get_data(dataset_name,dataset_list,tokenizer, args)
    return train_dataset, valid_dataset