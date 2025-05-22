import re
import random
import torch
from datasets import load_dataset,concatenate_datasets 
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
    elif datasetName == "tasksource/mmlu":
        option = examples[key[1]]
        examples["label"] = examples[key[2]]
    option_choices = insertOption(option)
    examples["target"] = f"{chr(int(examples['label'])+ord('A'))}. {option[int(examples['label'])]}" #option[int(examples["label"])] #chr(int(examples["label"])+ord('A'))#
    examples["input"] = fewshotPrompt + f"Question: {examples[key[0]]} \n" + (" \n").join(option_choices)+"\nWithout any explanation, select the best answer.\nAnswer:"
    examples["input_no_few"] = f"Question: {examples[key[0]]} \n" + (" \n").join(option_choices)+"\nWithout any explanation, select the best answer.\nAnswer:"
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
        #dataset = dataset.map(adjustAutoRegressive,num_proc=args.num_process)
        dataset = dataset.map(adjustAutoRegressive)

    return dataset

def get_data(dataset_name,dataset_list, tokenizer, args, modified_evaluation_dataset=False):
    # Load train and validation datasets
    #print("*"*30)
    #print("Loading Dataset")
    if isinstance(dataset_name,list):
        if dataset_name[0] == "tasksource/mmlu":
            traindata = load_dataset(dataset_name[0],dataset_name[1], split="test")#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets") 
            valdata = load_dataset(dataset_name[0], dataset_name[1],split="validation")#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets")  
        elif dataset_name[0] == "tasksource/bigbench":
            traindata = load_dataset(dataset_name[0],dataset_name[1], split="train",trust_remote_code=True)#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets") 
            valdata = load_dataset(dataset_name[0], dataset_name[1],split="validation",trust_remote_code=True)#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets") 
    else:
        if dataset_name == "EleutherAI/truthful_qa_mc":
            traindata = load_dataset(dataset_name, split="validation")#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets") 
            valdata = load_dataset(dataset_name, split="validation")#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets") 
        else:
            traindata = load_dataset(dataset_name, split="train")#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets") 
            valdata = load_dataset(dataset_name, split="validation")#,cache_dir="/gpfs/u/home/LLMG/LLMGbhnd/scratch/huggingface-cache/datasets") 
    valdata= valdata.shuffle(seed=args.seed)
    traindata.cleanup_cache_files()
    valdata.cleanup_cache_files()
    if isinstance(dataset_name,list):
        traindata = modifyDataset(traindata,dataset_list[dataset_name[0]]["keys"], "",dataset_list[dataset_name[0]]["prefixes"],dataset_name[0],args)
        valdata = modifyDataset(valdata,dataset_list[dataset_name[0]]["keys"],dataset_list[dataset_name[0]]["fewshot_prompt"],dataset_list[dataset_name[0]]["prefixes"],dataset_name[0],args)
    else:
        traindata = modifyDataset(traindata,dataset_list[dataset_name]["keys"], "",dataset_list[dataset_name]["prefixes"],dataset_name,args)
        valdata = modifyDataset(valdata,dataset_list[dataset_name]["keys"],dataset_list[dataset_name]["fewshot_prompt"],dataset_list[dataset_name]["prefixes"],dataset_name,args)
    
    #print("*"*30)
    #print("Generating Samples")
    # Generate samples from training set
    trainloader = []
    if args.do_train_both:
        num_dataset = int(args.nsamples /2)
    else:
        num_dataset = args.nsamples
    letter = [chr(65+i) for i in range(26)]
    i = 0
    for num in range(num_dataset):
        i = random.randint(0, len(traindata) - 1)
        if dataset_name[-1] == "abstract_narrative_understanding":
            while traindata[i]['target'].split(".")[0] not in letter:
                i = random.randint(0, len(traindata) - 1) 
        #if traindata[i]['target'] not in traindata[i]['target']
        text = traindata[i]['input_no_few'] +traindata[i]['target']#.split(".")[0]
        trainenc = tokenizer(text, return_tensors='pt',padding='max_length', max_length=args.seqlen,truncation=True)
        #trainenc = tokenizer(traindata[i]['input'], return_tensors='pt',padding=True, truncation=True)
        #label = tokenizer(, return_tensors='pt',padding='max_length', max_length=80,truncation=True)
        #i = trainenc.input_ids.shape[1]#random.randint(0, trainenc.input_ids.shape[1] - args.seqlen - 1)
        inp = trainenc.input_ids 
        atten = trainenc.attention_mask 
        #tar = inp.clone(
        #tar = label.input_ids#torch.cat((tar, label.input_ids), 1)
        tar = inp.clone()  # Shift left
        tar[tar == tokenizer.pad_token_id] = -100
        label = tokenizer(traindata[i]['target'].split(".")[0],return_tensors='pt').input_ids


        
        #inp = torch.where(inp != 32000, inp, -100)
        #tar = torch.where(tar != 32000, tar, -100)
        #tar[:, :i] = -100
        #inp = torch.nn.functional.pad(inp, (inp.size(1), args.seqlen - inp.size(1)), value=-100)
        #tar = torch.nn.functional.pad(tar, (tar.size(1), args.seqlen - tar.size(1)), value=-100)
        trainloader.append((inp,atten, tar,label))
    if modified_evaluation_dataset:
        # Check if validation dataset needs more samples
        if len(valdata) < args.evaluation_size:
            needed_samples = args.evaluation_size - len(valdata)
            extra_samples = traindata.shuffle(seed=args.seed).select(range(min(needed_samples, len(traindata))))
            valdata = concatenate_datasets([valdata, extra_samples]) 

    valloader = getDataLoader(tokenizerGiven=tokenizer, dataset=valdata, args=args)
    return trainloader, (valdata,valloader)

def getData(tokenizer,dataset_list, dataset_name, args,modified_evaluation_dataset=False):
    if args.do_train_both:
        train_dataset = []
        valid_dataset = []
        for dataset in dataset_name:
            train, _ = get_data(dataset,dataset_list,tokenizer, args,modified_evaluation_dataset)
            train_dataset += train
    else:
        train_dataset, valid_dataset = get_data(dataset_name,dataset_list,tokenizer, args,modified_evaluation_dataset)
    return train_dataset, valid_dataset