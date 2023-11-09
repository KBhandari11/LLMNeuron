from tqdm import tqdm 
import torch
from torch.optim import AdamW
from torch.optim.adamw import AdamW
from transformers import set_seed, TrainingArguments
from utils.dataset import load_dataset, modifyDataset
from utils.tokenizer import getDataLoader
from utils.models import getModel, LlamaModel
from pynvml import *
from accelerate import Accelerator
def print_gpu_utilization(idx):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(idx)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
def finetune(args,dataset_name,dataset_list):
    # Initialize accelerator
    torch.cuda.empty_cache()
    set_seed(0)
    if args.model_type == "llama":
        model, tokenizer = LlamaModel()
    else:
        model, tokenizer = getModel(args)
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        gradient_checkpointing=True,
        fp16=True,
        output_dir=f"checkpoints/{args.model_type}/finetune/{dataset_name.split('/')[-1]}/"
        )
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    traindata = load_dataset(dataset_name, split="train") 
    traindata = modifyDataset(traindata,dataset_list[dataset_name]["keys"], "",dataset_list[dataset_name]["prefixes"],dataset_name,args)
    #traindata.cleanup_cache_files()
    train_dataloader = getDataLoader(tokenizerGiven = tokenizer,dataset = traindata,args=args)
    #model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)    
    # Now we train the model

    #accelerator = Accelerator()
    #model.to(accelerator.device)

    #train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)
    print_gpu_utilization(6)
    print_gpu_utilization(7)
    for epoch in range(args.num_epochs):
        # We only enable the progress bar on the main process to avoid having 8 progress bars.
        progress_bar = tqdm(range(len(train_dataloader)))
        progress_bar.set_description(f"Epoch: {epoch}")
        model.train()
        for step, data in enumerate(train_dataloader):
            print(data)
            print_gpu_utilization(6)
            print_gpu_utilization(7)
            outputs = model(input_ids = data["input_ids"], 
                            attention_mask = data["attention_mask"],
                            labels = data["labels"]
            )
            '''outputs = model(input_ids = data["input_ids"].squeeze(0).to(args.device), 
                            attention_mask = data["attention_mask"].squeeze(0).to(args.device),
                            labels = data["labels"].to(args.device)
            )'''
            #outputs = model(**batch)
            loss = outputs.loss#/ training_args.gradient_accumulation_steps
            print(loss)
            progress_bar.set_postfix({'loss': loss.item()})
            progress_bar.update(1)
            loss.backward()
            # accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                

    #model.save_pretrained(f"checkpoints/{args.model_type}/finetune/{dataset_name.split('/')[-1]}/")
    #accelerator.save_state(f"checkpoints/{args.model_type}/finetune/{dataset_name.split('/')[-1]}/")

    del(model)