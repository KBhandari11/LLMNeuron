import pandas as pd
import json 
import math
import numpy as np
import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt
# Import necessary libraries
mpl.rcParams.update(mpl.rcParamsDefault)
#import itertools

from utils.bag_of_words.network_property import get_network_property
from utils.bag_of_words.permutation_test import permutation_test
from utils.bag_of_words.bipartite_multipartite_projection import create_plot_bog_modules,get_projection 
from utils.bag_of_words.sparsification import spectral_sparsification

def set_random_seed(seed=0):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


from utils.bag_of_words.skill_dataset import *

with open("result/dataNeuropsychologicalDomainsCluster.json", 'r') as openfile:
    # Reading from json file
    dataset_community= json.load(openfile)
with open("result/dataMultidisciplinaryCognitiveSkillsFrameworkRestrict.json", 'r') as openfile:
    #with open("result/dataCategory.json", 'r') as openfile:
    # Reading from json file
    dataCategory = json.load(openfile)
cognitive_skills_community = {
                    "cognitive_process_memory":[ 
                        "sustained_attention", "selective_attention", "divided_attention", "vigilance_attention","attention_shifting",
                        "processing_speed", "visual_processing_speed", "auditory_processing_speed",
                        "prospective_memory", "working_memory", "episodic_memory", "semantic_memory", "procedural_memory", "iconic_memory", "echoic_memory", "spatial_memory"],
                    "executive_function":[ 
                        "planning", "organization", "goal_setting","time_management", 
                        "problem_solving", "mental_flexibility", "strategic_thinking","adaptability",
                        "impulse_control", "decision_making","emotional_regulation","risk_assessment",
                        "abstract_thinking", "reasoning"," concept_formation", "cognitive_flexibility", "creativity"],
                    "language_communication":[
                         "expressive_language", "receptive_language", "naming", "fluency", "comprehension", "repetition", "reading", "writing", 
                         "pragmatics", "discourse_ability", "expressive_language", "receptive_language", "linguistic_analysis", "narrative_skills"],
                    "social_cognition":
                        ["recognition_of_social_cues", "theory_of_mind", "empathy", "social_judgment","intercultural_competence","conflict_resolution","self_awareness","relationship_management"]
}
all_skill_label = []
for func, skill_list in cognitive_skills_community.items():
    all_skill_label += skill_list


AB_dataset_skill, skill_label = create_plot_bog_skills(dataCategory, dataset_list, plot=False)
sparsity_ratio  = "25"
alpha = 0.01
data = {"model":[],"sparsity_ratio":[],"pruner_style":[],"metric":[],"module_statistics":[],"module_pvalue":[],"module_original":[],"module_random":[],"skill_statistics":[],"skill_pvalue":[],"skill_original":[],"skill_random":[]}
for model in ["llama","llama_chat","vicuna"]:
    print(model)
    for sparsity_ratio in ["25"]:
        for pruner_style in ["block","channel"]:
            print("\t",pruner_style,flush=True)
            original_network_list_skills = []
            original_network_list_modules = []
            print("\t\t",end="")
            for random_seed in range(10):
                set_random_seed(random_seed*12312312)
                print(random_seed,end=", ",flush=True)
                if model == "llama":
                    BC_dataset_modules, module_label = create_plot_bog_modules(llama_distribution,llama_original, dataset_list,pruner_style=pruner_style, pruner_ratio=sparsity_ratio,norm="|W|_0",plot=False, alpha=alpha)
                elif model == "llama_chat":
                    BC_dataset_modules, module_label = create_plot_bog_modules(llama_chat_distribution,llama_chat_original, dataset_list,pruner_style=pruner_style, pruner_ratio=sparsity_ratio,norm="|W|_0",plot=False, alpha=alpha)
                elif model == "vicuna":
                    BC_dataset_modules, module_label = create_plot_bog_modules(vicuna_distribution,vicuna_original, dataset_list,pruner_style=pruner_style, pruner_ratio=sparsity_ratio,norm="|W|_0",plot=False, alpha=alpha)
                
                #data = {n:[] for n in ['average_degree', 'average_cluster', 'density', 'global_efficiency', 'assortativity_coefficient', 'diameter', 'transitivity', 'avg_shorted_path_length', 'avg_betweenness_centrality', 'avg_eigen_centrality', 'spectral_radius']}
                AC_skill_modules =  np.dot(AB_dataset_skill.T,BC_dataset_modules)
                sparse_network = spectral_sparsification(AC_skill_modules, alpha)
                A_skill_skill,A_modules_modules  = get_projection(sparse_network, plot_projection= False)
                
                
                G_skill, _ = get_network_property(A_skill_skill,skill_label,skill_label )
                G_modules, _ = get_network_property(A_modules_modules,module_label,module_label )
                G_skill.remove_edges_from(nx.selfloop_edges(G_skill))
                G_modules.remove_edges_from(nx.selfloop_edges(G_modules)) 
                original_network_list_skills.append(G_skill)
                original_network_list_modules.append(G_modules)
            print()
            module_p_values = permutation_test(original_network_list_modules)
            skill_p_values = permutation_test(original_network_list_skills)
            for (p_value_metric, module_result),(p_value_metric, skill_result) in zip(module_p_values.items(),skill_p_values.items()):
                data["model"].append(model)
                data["sparsity_ratio"].append(sparsity_ratio)
                data["pruner_style"].append(pruner_style)
                data["metric"].append(p_value_metric)
                data["module_statistics"].append(module_result['statistics'])
                data["module_pvalue"].append(module_result['pvalue'])
                data["module_original"].append(module_result['original'])
                data["module_random"].append(module_result['random'])
                data["skill_statistics"].append(skill_result['statistics'])
                data["skill_pvalue"].append(skill_result['pvalue'])
                data["skill_original"].append(skill_result['original'])
                data["skill_random"].append(skill_result['random'])
            print("Saved to file:",flush=True)
        df = pd.DataFrame.from_dict(data)
        df.to_csv("result/compare_projection_random.csv",index=False)
#A_skill_modules,skill_label,module_label = combine_bipartite(dataset_list=dataset_list, skills=dataCategory, distribution=llama_distribution,original_distribution=llama_original, pruner_style="channel",pruner_ratio="25",norm="|W|_0", alpha=85, plot=True)
