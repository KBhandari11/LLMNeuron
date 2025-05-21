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

def mcq_token_index(tokenizer, mcq_choices):
    if len(mcq_choices) == 0:  
        token_ids = {chr(letter_index+ord('A')): tokenizer.convert_tokens_to_ids(chr(letter_index+ord('A'))) for letter_index in range(10)}
    else:
        token_ids = {chr(letter_index+ord('A')): tokenizer.convert_tokens_to_ids(chr(letter_index+ord('A'))) for letter_index in range(len(mcq_choices))}
    return token_ids

def checkAnswer(generated, true):
    '''strip_generated = ""
    for gen in generated:
        if gen.startswith('▁'):
            strip_generated = [*gen][-1].upper()
            break 
    '''
    #print("\nGenerated: ",strip_generated, "|\nTrue: ",true)

    if generated.lower() == true.lower():
        return generated, 1
    else:
        return generated, 0

def computeLogits(logits,lbls_map, choices_index):
    #print("Logits",outputs.logits.shape)
    #print("Logits",outputs.logits[0, -2].shape)
    if isinstance(logits,torch.Tensor):
        #print("Logits",logits.shape,logits,flush =True)
        logits = logits[-1, :]#logits[3,:]
        #print("After Logits",logits.shape,logits,flush =True)
    else:
        #print("Here:",torch.stack(logits)[0, -1].shape,file=sys.stderr)
        #print("Logits",(len(logits),logits[0].shape),logits,flush =True)
        logits = torch.stack(logits)[0, -1]

    probs = logits.softmax(dim=-1)
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
        ) if k in choices_index.keys()
    ]
    return logprobs_dict

def getAnswer(tokenized_inputs, model, tokenizer, vocab_map,choices_index, args):
    with torch.no_grad():
        #print("Input",tokenized_inputs["input_ids"].shape)
        outputs = model.generate(
            input_ids=tokenized_inputs["input_ids"].to(args.device),
            attention_mask=tokenized_inputs["attention_mask"].to(args.device),
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True
            #return_legacy_cache=True
        )
        # Decode and print the generated output
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        input_text = tokenizer.decode(tokenized_inputs["input_ids"][0], skip_special_tokens=True)
        #print("Output: ",generated_text.replace(input_text,""), flush=True)
        #print("Input Text:", input_text)
        #print("\tGenerated Output:", generated_text)
        generate = generated_text.replace(input_text,"")
        '''possible_token_ids = [token for _, token in choices_index.items()]
        possible_token_ids = torch.tensor( possible_token_ids, device=args.device)
        outputs = model(input_ids=tokenized_inputs["input_ids"].to(args.device), attention_mask=tokenized_inputs["attention_mask"].to(args.device), return_dict=True)
        logits = outputs.logits
        logits_for_possible_tokens = F.softmax(logits, dim=-1)[:,3, possible_token_ids] 
        #probs = torch.softmax(logits_for_possible_tokens, dim=-1)
        predicted_tokens = possible_token_ids[torch.argmax(logits_for_possible_tokens, dim=-1)]
        print(    tokenizer.batch_decode(possible_token_ids[torch.argmax(logits[:,0, possible_token_ids], dim=-1)]),
                  tokenizer.batch_decode(possible_token_ids[torch.argmax(logits[:,1, possible_token_ids], dim=-1)]),
                  tokenizer.batch_decode(possible_token_ids[torch.argmax(logits[:,2, possible_token_ids], dim=-1)]),
                  tokenizer.batch_decode(possible_token_ids[torch.argmax(logits[:,3, possible_token_ids], dim=-1)]),
                  flush=True)
        return tokenizer.batch_decode(predicted_tokens)#'''
    return computeLogits(outputs.logits,vocab_map,choices_index),generate

def generateText(model, tokenizer, dataloader, args):
    model.eval()
    #model.to(args.device)
    accuracy = []
    accuracy_gen = []
    lbls_map = {v: k for k, v in tokenizer.get_vocab().items()}
    saveData = []
    j = 0
    for i, (input, data) in enumerate(zip(dataloader[0],dataloader[1])):
        #input,generated,true = getAnswer(tokenized_inputs= data, model=model, tokenizer=tokenizer, vocab_map= lbls_map , args=args)
        true =input["target"].split(".")[0]
        data["input_ids"],data["attention_mask"] = data["input_ids"],data["attention_mask"]#[:,0:4086]
        #print(data["input_ids"].shape)
        if "choices" in input:
            get_choices_option = "choices"
        elif "options" in input:
            get_choices_option = "options"
        elif "multiple_choice_targets" in input:
            get_choices_option = "multiple_choice_targets" 
        choices_index =  mcq_token_index(tokenizer, input[get_choices_option])
        mcq_sorted_logits, generated_text = getAnswer(tokenized_inputs=data, model=model, tokenizer=tokenizer, vocab_map= lbls_map, choices_index= choices_index, args=args)
        generated_logits_estimated, acc =  checkAnswer(mcq_sorted_logits[0], true)
        #print("\t\t",generated, true ,flush=True)
        #print('++'*100, flush=True)
        acc_gen = 1 if true in generated_text else 0 
        accuracy.append(acc)
        accuracy_gen.append(acc_gen)         
       
        #print([input["input"],generated,true,acc])
        saveData.append([input["input_no_few"],(generated_text, generated_logits_estimated),true,(acc_gen, acc)])
        if args.evaluation_size+j == i+1:
            break
    #print("Device: ",args.device, end=" -> ")
    #print_gpu_utilization(int(args.device.split(":")[-1]))
    del model 
    #print([(g,e,t,s) for _,(g,e),t,s in saveData],file=sys.stderr)
    return (sum(accuracy), len(accuracy),sum(accuracy)/len(accuracy)),(sum(accuracy_gen), len(accuracy_gen),sum(accuracy_gen)/len(accuracy_gen)),[(a,b) for _,a,b,s in saveData], saveData

def evaluate(model,tokenizer,testloader,args):
    # Evaluate ppl in no grad context to avoid updating the model
    (acc, total, mean),eval_gen,predicted, saveData = generateText(model, tokenizer, testloader,args)
    return (acc, total,mean),eval_gen,predicted,  saveData