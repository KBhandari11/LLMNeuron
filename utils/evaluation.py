import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pynvml import *
def print_gpu_utilization(idx):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(idx)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def checkAnswer(generated, true):
    strip_generated = ""
    for gen in generated:
        if gen.startswith('▁'):
            strip_generated = [*gen][-1].upper()
            break 

    #print("\nGenerated: ",strip_generated, "|\nTrue: ",true)

    if strip_generated == true:
        return strip_generated, 1
    else:
        return strip_generated, 0

def computeLogits(outputs,lbls_map, tokenizer):
    logits = outputs.logits[0, -1]
    probs = logits.softmax(dim=-1)
    preds = F.softmax(outputs.logits, dim=-1).argmax(dim=-1)
    print("Preds",preds)
    print("Probs",outputs.logits.softmax(dim=-1).argmax(dim=-1))
    print("Probs |0| ",outputs.logits[0, -1].softmax(dim=-1).argmax(dim=-1))
    print("Result Preds: ",tokenizer.batch_decode(sequences=preds, skip_special_tokens=True)) 
    
    print("Result Prob: ",tokenizer.batch_decode(sequences=probs.argmax(), skip_special_tokens=True)) 
    logprobs_dict = {
        lbls_map[i]:
        np.log(probs[i].item()) for i in range(len(lbls_map))
    }
    # Reduce logprobs_dict to only keys with top 50 largest values
    logprobs_dict = [
        k for k, v in sorted(
            logprobs_dict.items(),
            key=lambda item: item[1],
            reverse=True
        )[:50]
    ]
    return logprobs_dict

def getAnswer(tokenized_inputs, model, tokenizer, vocab_map, args):
    with torch.no_grad():
        #print_gpu_utilization(0)
        #outputs = model(tokenized_inputs["input_ids"].to(args.device))
        outputs = model(input_ids =tokenized_inputs["input_ids"].to(args.device),attention_mask =tokenized_inputs["attention_mask"].to(args.device))
        #preds = F.softmax(outputs.logits, dim=-1).argmax(dim=-1)
        #print(tokenizer.batch_decode(sequences=preds, skip_special_tokens=True))
    return computeLogits(outputs,vocab_map,tokenizer) #generated_text,generated_answer,true_text

def generateText(model, tokenizer, dataloader,args):
    model.eval()
    #model.to(args.device)
    accuracy = []
    lbls_map = {v: k for k, v in tokenizer.get_vocab().items()}
    saveData = []
    j = 0
    for i, (input, data) in enumerate(zip(dataloader[0],dataloader[1])):
        #input,generated,true = getAnswer(tokenized_inputs= data, model=model, tokenizer=tokenizer, vocab_map= lbls_map , args=args)
        true =input["target"].split(".")[0]
        data["input_ids"],data["attention_mask"] = data["input_ids"][:,0:5000],data["attention_mask"][:,0:5000]
        #print(data["input_ids"].shape)
        logits = getAnswer(tokenized_inputs=data, model=model, tokenizer=tokenizer, vocab_map= lbls_map , args=args)
        generated, acc =  checkAnswer(logits, true)
        print(generated, acc)
        accuracy.append(acc)
        #print([input["input"],generated,true,acc])
        saveData.append([input["input_no_few"],generated,true,acc])
        if args.evaluation_size+j == i+1:
            break
    #print("Device: ",args.device, end=" -> ")
    #print_gpu_utilization(int(args.device.split(":")[-1]))
    del model 
    print([(a,b) for _,a,b,_ in saveData],file=sys.stderr)
    return (sum(accuracy), len(accuracy),sum(accuracy)/len(accuracy)), saveData

def evaluate(model,tokenizer,testloader,args):
    # Evaluate ppl in no grad context to avoid updating the model
    (acc, total, mean), saveData = generateText(model, tokenizer, testloader,args)
    return (acc, total,mean), saveData