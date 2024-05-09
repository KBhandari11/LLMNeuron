import json
from utils.bag_of_words.network_property import *
from utils.bag_of_words.permutation_test import *
from utils.bag_of_words.skill_dataset import *
from utils.bag_of_words.sparsification import spectral_sparsification

import json 
import math
import numpy as np
import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt
# Import necessary libraries
mpl.rcParams.update(mpl.rcParamsDefault)
#import itertools
from utils.bag_of_words.skill_dataset import *
from utils.bag_of_words.network_property import *
from utils.bag_of_words.dataset_modules import *
from utils.bag_of_words.bipartite_multipartite_projection import *

def l_0_norm(vector):
    count = 0
    total = 0
    for element in vector:
        for sub_element in element:
            if sub_element != 0:
                count += 1
    return count
def take_average(dict):
    data = dict["0"]
    iterations_block = list(["0","1","2","3","4"])
    iterations_channel = list(["0","1","2","3","4"])
    #for style , iterations in zip (["block","channel","block_random","channel_random"],[iterations_block,iterations_channel,iterations_block,iterations_channel]):
    for style , iterations in zip (["block","channel"],[iterations_block,iterations_channel,iterations_block,iterations_channel]):
        for iter in iterations:
            if iter == "0":
                continue
            for ratio in dict[iter][style]:
                for dataset in dict[iter][style][ratio]:
                    for norm in dict[iter][style][ratio][dataset]:
                        value = np.array(dict[iter][style][ratio][dataset][norm])
                        if len( value.shape) != 1:
                            shape_model = value.shape
                        data[style][ratio][dataset][norm]= (np.array(data[style][ratio][dataset][norm])+value)
                        if iter == iterations[-1]:
                            data[style][ratio][dataset][norm] = data[style][ratio][dataset][norm]/len(iterations)
    return data, shape_model
def strip(name):
    name = name.split("/")[-1]
    name = name.split("_")[0]
    return name 

with open("result/distribution_llama_7b.json", 'r') as openfile:
    # Reading from json file
    llama_distribution = json.load(openfile)
with open("result/distribution_vicuna_7b.json", 'r') as openfile:
    # Reading from json file
    vicuna_distribution = json.load(openfile)
with open("result/distribution_llama_7b-chat.json", 'r') as openfile:
    # Reading from json file
    llama_chat_distribution= json.load(openfile)
def loop_over(dict):
    if isinstance(dict, list):
        print("end")
    else: 
        print(dict.keys())
        for keys in dict:
            loop_over(dict[keys])
def get_dataset_list(dataset_list):
    dataname = []
    for data in dataset_list:
        if "subset" not in dataset_list[data].keys():
            dataname.append(data)
        else:
            for subset in dataset_list[data]["subset"]:
                dataname.append(subset)
    return dataname
    #Dataset List
with open("/home/bhandk/MLNeuron/dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_list = json.load(openfile)
#Original Distribution
with open("result/original_distribution_vicuna_7b.json", 'r') as openfile:
    vicuna_original = json.load(openfile)
with open("result/original_distribution_llama_7b.json", 'r') as openfile:
    # Reading from json file
    llama_original = json.load(openfile)
with open("result/original_distribution_llama_7b-chat.json", 'r') as openfile:
    # Reading from json file
    llama_chat_original = json.load(openfile)
#Pruned Distribution
with open("result/distribution_llama_7b.json", 'r') as openfile:
    # Reading from json file
    llama_distribution = json.load(openfile)
with open("result/distribution_vicuna_7b.json", 'r') as openfile:
    # Reading from json file
    vicuna_distribution = json.load(openfile)
with open("result/distribution_llama_7b-chat.json", 'r') as openfile:
    # Reading from json file
    llama_chat_distribution= json.load(openfile)
with open("result/dataNeuropsychologicalDomainsCluster.json", 'r') as openfile:
    # Reading from json file
    dataset_community= json.load(openfile)

dataset_list = get_dataset_list(dataset_list)
llama_distribution, model_shape = take_average(llama_distribution)
vicuna_distribution, model_shape = take_average(vicuna_distribution)
llama_chat_distribution, model_shape = take_average(llama_chat_distribution)


with open("result/dataMultidisciplinaryCognitiveSkillsFrameworkRestrict.json", 'r') as openfile:
    #with open("result/dataCategory.json", 'r') as openfile:
    # Reading from json file
    dataCategory = json.load(openfile)

dataCategory1 = dataCategory#filterData(dataCategory, 1.0)#0.4
A_dataset_skill, skills = create_plot_bog_skills(dataCategory1, dataset_list, plot=False)
AB_dataset_skill, skill_label = create_plot_bog_skills(dataCategory1, dataset_list, plot=False)


for sparsity_ratio  in ["15","20","30","40"]:
    for pruner_style in ["block","channel"]:
        BC_dataset_modules, module_label = create_plot_bog_modules(llama_distribution,llama_original, dataset_list,pruner_style=pruner_style, pruner_ratio=sparsity_ratio,norm="|W|_0",plot=False, alpha=None)
        data = {n:[] for n in ['average_degree','average_cluster', 'num_community', 'modularity', 'density', 'alpha']}
        for alpha in [0.01,0.05,0.1,0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.5]:
            #AC_skill_modules =  np.dot(AB_dataset_skill.T,BC_dataset_modules)
            #A_skill_skill,A_modules_modules  = get_projection(A_skill_modules, plot_projection= False)
            original_network_list = []
            for _ in range(100):
                sparse_network = spectral_sparsification(BC_dataset_modules, alpha)
                G, _ = get_network_property(sparse_network,dataset_list,module_label )
                original_network_list.append(G)
            module_p_values = permutation_test(original_network_list)
            #_, property = get_network_property(AC_skill_modules,skill_label,module_label )
            #A_dataset_modules,module_label = create_plot_bog_modules(llama_distribution,llama_original, dataset_list,pruner_style="channel", pruner_ratio=sparsity_ratio,norm="|W|_0",plot=False, alpha=alpha)
            #_, property = get_network_property(A_dataset_modules,dataset_list,module_label )
            for p_value_metric, p_value in module_p_values.items():
                print(f"Model: Llama-7b | Sparsity ratio: {sparsity_ratio} | Pruner Style: {pruner_style} | Alpha: {alpha}| Metric: {p_value_metric} | Average Value(org): {p_value[2]} | Average Value(shuffle): {p_value[3]} |  Stats: {p_value[1]} |  P-Value: {p_value[0]}", flush=True)