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
from utils.bag_of_words.skill_dataset import *
from utils.bag_of_words.network_property import *
from utils.bag_of_words.dataset_modules import *
from utils.bag_of_words.permutation_test import *
from utils.bag_of_words.bipartite_multipartite_projection import *
from utils.bag_of_words.sparsification import spectral_sparsification
from sklearn.metrics import jaccard_score, normalized_mutual_info_score, rand_score,adjusted_rand_score, adjusted_mutual_info_score,mutual_info_score

def add_isolated(label, all_skill_label):
    new_label = {}
    total_comm = list(set(label.values()))
    for node in all_skill_label:
        if node in label:
            new_label[node] = label[node]
        else:
            new_label[node] = len(total_comm) +10
    return new_label
def get_community_for_alpha(dataCategory, dataset_list, distribution, original, pruner_style="block", sparsity_ratio="15",alpha1=None, alpha2=None, random_seed=True, modules_vs_modules=True):
    AB_dataset_skill, skill_label = create_plot_bog_skills(dataCategory, dataset_list, plot=False)
    BC_dataset_modules, module_label = create_plot_bog_modules(distribution,original, dataset_list,pruner_style=pruner_style, pruner_ratio=sparsity_ratio,norm="|W|_0",plot=False, alpha=alpha1, random_seed=random_seed)
    A_skill_modules =  np.dot(AB_dataset_skill.T,BC_dataset_modules)
    sparse_network = spectral_sparsification(A_skill_modules, alpha2, random_seed=random_seed)

    if modules_vs_modules:
        _ ,A_modules_modules  = get_projection(sparse_network, plot_projection= False)
        G, network_property = get_network_property(A_modules_modules,module_label,module_label )
    else:
        A_skills_skills , _ = get_projection(sparse_network, plot_projection= False)
        G, network_property = get_network_property(A_skills_skills,skill_label,skill_label )
    return G, network_property, (sparse_network,skill_label,module_label)



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
with open("./dataset_info.json", 'r') as openfile:
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

def set_random_seed(seed=0):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed()
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




distribution_dist = [llama_distribution,llama_chat_distribution,vicuna_distribution]
original_dist = [llama_original,llama_chat_original,vicuna_original]
pruner_style ="block"
def add_isolated(label, all_skill_label):
    new_label = {}
    total_comm = list(set(label.values()))
    for node in all_skill_label:
        if node in label:
            new_label[node] = label[node]
        else:
            new_label[node] = len(total_comm)
            total_comm.append(len(total_comm))
    return new_label
def get_ground_truth(all_node, cognitive_skills_community):
    ground_truth = {}
    for node in all_node:
        for comm_idx, comm in enumerate(cognitive_skills_community):
            if node in cognitive_skills_community[comm]:
                ground_truth[node] = comm_idx
    return ground_truth
cognitive_function_index_dict= get_ground_truth(all_skill_label, cognitive_skills_community)
cognitive_function_partition = np.array([cognitive_function_index_dict[skill] for skill in all_skill_label])
data = {"pruner_style":[],"model":[],"sparsity_ratio":[],"jaccard_index":[], "nmi":[], "rand_score":[], "adjusted_rand_score":[],"community":[],"cognitive_function":[]}
for pruner_style in ["block","channel"]:
    for model_idx ,model in enumerate(["llama","llama_chat","vicuna"]):
        for sparsity_ratio1 in ["3","15","20","25","30","35","40"]:
            G, property_1, _ = get_community_for_alpha(dataCategory, dataset_list, distribution_dist[model_idx], original_dist[model_idx], pruner_style=pruner_style, sparsity_ratio=sparsity_ratio1,alpha1=0.01,alpha2=0.01, modules_vs_modules=False)
            partition1 = add_isolated(property_1["partition"],all_skill_label)
            partition1 = np.array([comm for  _, comm in partition1.items()])

            print(partition1)
            print(cognitive_function_partition)
            
            jaccard = jaccard_score(cognitive_function_partition,partition1, average="micro")
            nmi = normalized_mutual_info_score(cognitive_function_partition,partition1)
            adjust_rand_score = adjusted_rand_score(cognitive_function_partition,partition1)
            rand_scores = rand_score(cognitive_function_partition,partition1)
            #jaccard = jaccard_score(np.array([comm for  _, comm in property_1["partition"].items()]), np.array([comm for  _, comm in property_2["partition"].items()]), average=None)
            #print(sparsity_ratio1,sparsity_ratio2,jaccard)
            data["model"].append(model)
            data["pruner_style"].append(pruner_style)
            data["sparsity_ratio"].append(sparsity_ratio1)
            data["jaccard_index"].append(jaccard)
            data["nmi"].append(nmi)
            data["rand_score"].append(rand_scores)
            data["adjusted_rand_score"].append(adjust_rand_score)
            data["community"].append(partition1)
            data["cognitive_function"].append(cognitive_function_partition)
data1 = pd.DataFrame(data)
data1.to_csv("result/compare_cog_functions_skills_community.csv")