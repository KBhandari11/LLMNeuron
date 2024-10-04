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
import collections
from scipy.stats import chi2_contingency, entropy
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

dataset_list = get_dataset_list(dataset_list)
llama_distribution, model_shape = take_average(llama_distribution)
vicuna_distribution, model_shape = take_average(vicuna_distribution)
llama_chat_distribution, model_shape = take_average(llama_chat_distribution)


from utils.bag_of_words.skill_dataset import *
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


def add_isolated(label, all_label):
    new_label = {}
    total_comm = list(set(label.values()))
    for node in all_label:
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

distribution_dist = [llama_distribution,llama_chat_distribution,vicuna_distribution]
original_dist = [llama_original,llama_chat_original,vicuna_original]
pruner_style ="block"
def get_ground_truth(all_node, cognitive_skills_community):
    ground_truth = {}
    for node in all_node:
        for comm_idx, comm in enumerate(cognitive_skills_community):
            if node in cognitive_skills_community[comm]:
                ground_truth[node] = comm_idx
    return ground_truth
def get_skills_shared_by_modules(G_module_module,community_detected,skills_modules,skills_label, module_label, all_skill_label):
    community = {}
    G_skill_module = create_biparitite(skills_modules,skills_label, module_label)
    for comm in community_detected:
        H_module_module = G_module_module.subgraph(community_detected[comm])
        shared_skills = []
        for u,v in list(H_module_module.edges):
            u_skills = set(G_skill_module.neighbors(u))
            v_skills = set(G_skill_module.neighbors(v))
            shared_skills += list(u_skills.intersection(v_skills))
        freq_skills = collections.Counter(shared_skills)
        community[comm] = [freq_skills[skill] if skill in freq_skills else 0 for skill in all_skill_label]
    return community
def get_modules_shared_by_skills(G_skills_skills,community_detected,skills_modules,skills_label, module_label, all_label):
    community = {}
    G_skill_module = create_biparitite(skills_modules,skills_label, module_label)
    for comm in community_detected:
        H_skills_skills = G_skills_skills.subgraph(community_detected[comm])
        shared_modules = []
        for u,v in list(H_skills_skills.edges):
            u_modules = set(G_skill_module.neighbors(u))
            v_modules = set(G_skill_module.neighbors(v))
            shared_modules += list(u_modules.intersection(v_modules))
        freq_modules = collections.Counter(shared_modules)
        community[comm] = [freq_modules[skill] if skill in freq_modules else 0 for skill in all_label]
    return community
def calculate_kl_divergence(community_freq, epsilon=1e-16):
    row = []
    for comm1 in community_freq:
        distribution_p = np.array(community_freq[comm1]) / sum(community_freq[comm1])
        smoothed_p = np.where(distribution_p == 0, epsilon, distribution_p)
        col = []
        for comm2 in community_freq:
            distribution_q = np.array(community_freq[comm2]) / sum(community_freq[comm2])
            smoothed_q = np.where(distribution_q == 0, epsilon, distribution_q)
            col.append(entropy(smoothed_p, smoothed_q))
        row.append(col)
    return np.array(row)
def create_frequency_skills(dataCategory, all_skill_label):
    freq ={}
    for data, skills in dataCategory.items():
        freq_skills = collections.Counter(skills)
        freq[data] = [freq_skills[skill] if skill in freq_skills else 0 for skill in all_skill_label]
    return freq
def check_kl_divergence(community_detected_frequency, sample_dict):
    comm_list = list(community_detected_frequency.keys())
    distributions = np.array([freq for _, freq in community_detected_frequency.items()],dtype='float64')
    data = {comm:[] for comm in community_detected_frequency}
    for samp in sample_dict:
        sample = np.array(sample_dict[samp],dtype='float64')
        
        # Normalize the distributions and sample if not already normalized
        distributions /= distributions.sum(axis=1, keepdims=True)
        sample /= sample.sum()

        # Compute Kullback-Leibler Divergence from the sample to each distribution
        kl_divergences = [entropy(sample, dist) for dist in distributions]

        # Find the distribution with the minimum KL divergence
        closest_distribution_index = np.argmin(kl_divergences)
        data[comm_list[closest_distribution_index]].append(samp)

    return data


def sum_cogn_function(fequencies, index):
    data = []
    for cog_functions in np.unique(index):
        data.append(sum(np.take(fequencies,  np.where(index==cog_functions)[0], 0)))
    return np.array(data)
random_seed = True
modules_vs_modules = True # if comparing skills vs skills
if not modules_vs_modules:
    modules=["attn.q", "attn.k", "attn.v", "attn.o","gate","mlp.up", "mlp.down"]
    all_label_modules=[ str(i)+"_"+m  for i in range(3,31) for m in modules]
else:
    all_label_skills = all_skill_label
    ground_truth = get_ground_truth(all_label_skills,cognitive_skills_community)
    ground_truth = np.array([comm for  _, comm in ground_truth.items()])
#plot_community_model(G,property["partition"])

data = {"pruner_style":[],"model":[],"sparsity_ratio":[], "statistic":[], "p":[], "dof":[], "entropy":[],"data":[]}
for pruner_style in ["block","channel"]:
    for model_idx ,model in enumerate(["llama","llama_chat","vicuna"]):
        for sparsity_ratio in ["3","15","20","25","30","35","40"]:
            G, property_1, (skills_modules, skill_label, module_label) = get_community_for_alpha(dataCategory, dataset_list, distribution_dist[model_idx], original_dist[model_idx], pruner_style=pruner_style, sparsity_ratio=sparsity_ratio,alpha1=0.01,alpha2=0.01,random_seed=random_seed, modules_vs_modules=modules_vs_modules)
            if modules_vs_modules:
                community_detected_frequency = get_skills_shared_by_modules(G,property_1["community"],skills_modules,skill_label, module_label, all_label_skills)
            else:
                community_detected_frequency = get_modules_shared_by_skills(G,property_1["community"],skills_modules,skill_label, module_label, all_label_modules)
            
            '''for idx, c in enumerate(community_detected_frequency):
                print("Given",c, community_detected_frequency[c])
                print("Expected",c, res.expected_freq[idx])
            print("*"*100)'''
        
            entropy_matrix = calculate_kl_divergence(community_detected_frequency)
            feq_skills = np.array([freq for _, freq in community_detected_frequency.items()])
            feq_cog_function = np.array([sum_cogn_function(freq,ground_truth) for _, freq in community_detected_frequency.items()])
            feq = feq_cog_function
            res = chi2_contingency(feq)
            data["model"].append(model)
            data["pruner_style"].append(pruner_style)
            data["sparsity_ratio"].append(sparsity_ratio)
            data["statistic"].append(res.statistic)
            data["p"].append(res.pvalue)
            data["dof"].append(res.dof)
            data["entropy"].append(np.sum(entropy_matrix))
            data["data"].append(feq)

df = pd.DataFrame.from_dict(data)
df.to_csv("result/compare_projection_random.csv",index=False)
#A_skill_modules,skill_label,module_label = combine_bipartite(dataset_list=dataset_list, skills=dataCategory, distribution=llama_distribution,original_distribution=llama_original, pruner_style="channel",pruner_ratio="25",norm="|W|_0", alpha=85, plot=True)
