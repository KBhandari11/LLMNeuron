from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import  PreTrainedTokenizer 
import torch
import torch.nn.functional as func
from transformers import DataCollatorWithPadding
def collate_fn(batch):
  return {
      'input_ids': torch.stack([x['input_ids'] for x in batch]),
      'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
}
def createTokens(examples):
    '''
    Create Tokens for all the inputs. 
    keys = ["question","labels"] depending upon different variable. 
    '''
    results = {}
    #model_inputs  = tokenizer(examples["input"], return_tensors="pt", padding=True, truncation=False)
    model_inputs  = tokenizer(examples["input_no_few"], return_tensors="pt")
    #results["input_ids"] = model_inputs["input_ids"]
    #results["attention_mask"] = model_inputs["attention_mask"]
    #results["labels"]  = tokenizer(examples["label"]).input_ids
    #results["labels"]  = tokenizer(examples["label"]).input_ids
    #results['labels'] = torch.squeeze(func.one_hot(torch.tensor(examples["label"]), num_classes = len(examples["input"])))
    return model_inputs

def getDataLoader(tokenizerGiven: PreTrainedTokenizer,dataset: DataLoader,args:ArgumentParser):
    global tokenizer
    global max_length 
    global finetune
    max_length  = args.max_length
    tokenizer = tokenizerGiven
    finetune =  args.fine_tune

    #dataset = dataset.map(createTokens,batched=True,batch_size=args.batch_size,num_proc=args.num_process)
    dataset = dataset.map(createTokens,batched=False,num_proc=args.num_process)
    #dataset.cleanup_cache_files()
    #dataset.cleanup_cache_files()
    #dataset_new.set_format(type="torch",columns=["input_ids","attention_mask"])
    dataset.set_format(type="torch",columns=["input_ids","attention_mask"])
    #dataloader = DataLoader(dataset_new,num_workers=args.num_process)
    return dataset #dataloader