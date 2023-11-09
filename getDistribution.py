from utils.prune import find_layers
import torch 
import os
import numpy as np
from argparse import Namespace
import seaborn as sns
import matplotlib.pyplot as plt
from utils.models import *
from transformers import AutoModelForCausalLM
from utils.prune import check_sparsity
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json

def create_distribution_llm_pruner(model):
    distribution_F = []
    distribution_0 = []
    count = 0 
    total_params = 0 
    for layers in range(32):
        data_layer_F = []
        data_layer_0 = []
        for sub_layer in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            key_name = f'model.layers.{layers}.{sub_layer}.weight'
            if key_name in model.keys():
                W = model[key_name]
                count += (W==0).sum().item()
                total_params += W.numel()
                data_layer_F.append(torch.linalg.matrix_norm(W).item())#|W|_F norm
                data_layer_0.append((W.numel() - (W==0).sum().item()))#|W|_0 norm
            else:
                data_layer_F.append(0)
                data_layer_0.append(0)
        distribution_F.append(data_layer_F)
        distribution_0.append(data_layer_0)
    return float(count)/total_params, np.array(distribution_F),np.array(distribution_0)
def create_distribution(model):
    layers = model.model.layers
    distribution_F = []
    distribution_0 = []
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_values_F = []
        layer_values_0 = []
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()
            layer_values_F.append(torch.linalg.matrix_norm(W).item()) #|W|_F norm
            layer_values_0.append((W.numel() - (W==0).sum().item())) #|W|_0 norm
        distribution_F.append(layer_values_F)
        distribution_0.append(layer_values_0)
    return float(count)/total_params, np.array(distribution_F),np.array(distribution_0)

def get_distribution_wanda(args,dir,dataset_name,ratio):
    args.sparsity_ratio = ratio
    args.save_data = f'{dir}{ratio}/{dataset_name.split("/")[-1]}/'
    args.model = args.save_data
    model, _ = getModel(args,evalCond=True)
    sparsity , distribution_F,distribution_0 = create_distribution(model)
    del(model)
    return distribution_F,distribution_0 
def get_original(model_name):
    args = Namespace(save_data = "",model_type="llama",sparsity_ratio=0.0,model=model_name)
    model, _ = getModel(args,evalCond=True)
    _, distribution_F, distribution_0 = create_distribution(model)
    del(model)
    return distribution_F, distribution_0
def get_distribution_llm_pruner(dir,dataset_name,ratio):
    model = torch.load(f"{dir}{ratio}/{dataset_name}", map_location=torch.device('cpu'))
    sparsity_ratio, distribution_F, distribution_0 = create_distribution_llm_pruner(model) 
    print(f"Sparsity sanity check of LLM PRUNER {dir}{ratio}/{dataset_name}: {sparsity_ratio:.4f}")
    del(model)
    return distribution_F, distribution_0
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
def get_distribution_sparsity(dir,llm_pruner_cond=True):
    args = Namespace(save_data = "",model_type="llama",sparsity_ratio=0.0,model="")
    files_with_ratio = list(os.walk(dir))[0][1]
    w_f_ratio = {}
    w_0_ratio = {}
    for ratio in files_with_ratio:
        if not(ratio in ["15-chat_1","15_1"]):
            continue
        if llm_pruner_cond:
            dataset_list = list(os.walk(f"{dir}{ratio}/"))[0][2] 
        else:
            dataset_list = list(os.walk(f"{dir}{ratio}/"))[0][1] 
        if len(list(dataset_list)) == 0: #ignoring if the directory is empty
            continue
        w_f = {}
        w_0 = {}
        for dataset in dataset_list:
            if llm_pruner_cond:
                dataset_name = dataset.split("model_")[1].split(".")[0]
                w_f[dataset_name],w_0[dataset_name]= get_distribution_llm_pruner(dir,dataset,ratio)
            else:
                w_f[dataset],w_0[dataset]= get_distribution_wanda(args,dir,dataset,ratio)
        w_f_ratio[ratio] = w_f
        w_0_ratio[ratio] = w_0
    return w_f_ratio, w_0_ratio

def main():
    '''args = Namespace(save_data = "",model_type="llama",sparsity_ratio=0.0,model="")
    SAVE_WANDA = "checkpoints/llama"
    files = list(os.walk(SAVE_WANDA))[0][1]
    print(files)
    all_distribution = {"|W|_F":{},"|W|_0":{}}
    for ratio in files:
        dataset_list = list(os.walk(f"{SAVE_WANDA}/{ratio}"))[0][1] 
        print(list(dataset_list))
        if len(list(dataset_list)) == 0:
            continue
        all_distribution["|W|_F"][ratio] = {}
        all_distribution["|W|_0"][ratio] = {}
        for dataset in dataset_list:
            if not(dataset in all_distribution["|W|_F"][ratio].keys()):
                all_distribution["|W|_F"][ratio][dataset],all_distribution["|W|_0"][ratio][dataset]= get_distribution(args,dataset,ratio)
    all_distribution["|W|_F"]["original"],all_distribution["|W|_0"]["original"] = get_original(model_name="meta-llama/Llama-2-7b-hf")
    all_distribution["|W|_F"]["original-chat"],all_distribution["|W|_0"]["original-chat"] = get_original(model_name="meta-llama/Llama-2-7b-chat-hf")
    #with open("result/all_distribution["|W|_F"]_org.json", 'r') as openfile:
    #    # Reading from json file
    #    all_distribution["|W|_F"] = json.load(openfile)
    dataset_list = ["commonsense_qa","cais/mmlu","math_qa","EleutherAI/truthful_qa_mc","tasksource/bigbench","derek-thomas/ScienceQA"]
    style_dict = {}
    for style in ["block","channel","layer"]:
        ratio_dict = {}
        for ratio in ['15','15-chat','25','25-chat']:
            dataset_dict = {}
            for dataset_name in dataset_list:
                dataset_name = dataset_name.split("/")[-1] 
                dataset_dict[dataset_name]= get_llm_pruner(ratio,dataset_name,style)
            ratio_dict[ratio] = dataset_dict
        style_dict[style] = ratio_dict
    all_distribution["|W|_F"]["llm_pruner"] = style_dict
    json_object = json.dumps(all_distribution["|W|_F"], cls=NumpyEncoder)
    with open("result/all_distribution_org_pruner.json", "w") as outfile:
        outfile.write(json_object)'''
    SAVE = "/data/Kushal/MLNeuron/checkpoints/llama-7b/"
    all_distribution = {"|W|_F":{},"|W|_0":{}}
    for pruner_type in ["LLM_Pruner"]:
        dir = SAVE + pruner_type +"/"
        all_distribution["|W|_F"][pruner_type] = {}
        all_distribution["|W|_0"][pruner_type] = {}
        if pruner_type == "LLM_Pruner":
            files_with_type = list(os.walk(dir))[0][1]
            for style in files_with_type:
                if style != "layer":
                    all_distribution["|W|_F"][pruner_type][style],all_distribution["|W|_0"][pruner_type][style] = get_distribution_sparsity(dir+style+"/",llm_pruner_cond=True)
        else:
            all_distribution["|W|_F"][pruner_type],all_distribution["|W|_0"][pruner_type] = get_distribution_sparsity(dir,llm_pruner_cond=False)
            
    all_distribution["|W|_F"]["original"],all_distribution["|W|_0"]["original"] = get_original(model_name="meta-llama/Llama-2-7b-hf")
    all_distribution["|W|_F"]["original-chat"],all_distribution["|W|_0"]["original-chat"] = get_original(model_name="meta-llama/Llama-2-7b-chat-hf")
    json_object = json.dumps(all_distribution, cls=NumpyEncoder)
    with open("result/all_distribution_new_1.json", "w") as outfile:
        outfile.write(json_object)
if __name__ == "__main__":
    main()