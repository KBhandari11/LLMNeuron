import json
import sys
import torch 
import torch.nn as nn
import numpy as np
import argparse
from argparse import Namespace
from utils.dataset import getData
from utils.evaluation import evaluate

from transformers import LlamaTokenizer

sys.path.append("./LLM-Pruner")
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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


def set_random_seed(seed):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='base model name')
    parser.add_argument('--save_distribution_path', type=str, default="./result/original_distribution_llama_7b-chat.json", help='path to save the result')
    parser.add_argument('--device', type=str, default="cuda", help='GPU: cuda')
    args = parser.parse_args()

    with open("./dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_info_list = json.load(openfile)   

    
    dataset_list = get_dataset_list(dataset_info_list)
    value = {}
    value["distribution"]={}
    model = LlamaForCausalLM.from_pretrained(
                args.base_model,
                low_cpu_mem_usage=True #if args.torch_version >=1.9 else False,
            )
    model.half()
    model.to(args.device)
    value["distribution"]["|W|_F"],value["distribution"]["|W|_inf"], value["distribution"]["|W|_0"],sparsity = create_distribution_llm_pruner(model)
    del model
    torch.cuda.empty_cache()
    for dataset_name in dataset_list:
            print(dataset_name)
            distribution = {"|W|_F":None,"|W|_inf":None,"|W|_0":None,"Accuracy":None}
            tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
            model = LlamaForCausalLM.from_pretrained(
                args.base_model,
                low_cpu_mem_usage=True #if args.torch_version >=1.9 else False,
            )
            args_dataset = Namespace(save_data = "",do_train_both = False,nsamples=1,seqlen=500,model_type="llama",num_process=10,max_length=0,device='cpu',fine_tune=False)
            _, validation_dataset = getData(tokenizer,dataset_info_list, dataset_name, args_dataset)
            accuracy,_ = evaluate(model=model,tokenizer=tokenizer,testloader=validation_dataset,args=args)
            if isinstance(dataset_name,list):
                value[dataset_name[-1].split('/')[-1]] = accuracy
                print(dataset_name[-1].split('/')[-1], accuracy)
            else:
                value[dataset_name.split('/')[-1]] = accuracy
                print(dataset_name.split('/')[-1], accuracy)
    json_object = json.dumps(value, cls=NumpyEncoder)
    with open(args.save_distribution_path, "w") as outfile:
        outfile.write(json_object)