import random
import torch
import numpy as np

import json 
import math
import matplotlib as mpl
from matplotlib import rc

import networkx as nx
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
from math import floor


import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from utils.bag_of_words.network_property import get_network_property

from utils.bag_of_words.projection_community import create_projection_network,all_skill_label

def set_random_seed(seed):
    random.seed(seed)
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


def get_all_dataset_list(dataset_info_list, dataset_list):
    dataname = []
    for d in dataset_list:
        for data in dataset_info_list:
            if "subset" not in dataset_info_list[data].keys():
                if  data == d:
                    dataname.append(data)
                    continue
            else:
                if d in dataset_info_list[data]["subset"]:
                    dataname.append([data,d])
                    continue
    return dataname


def min_max(X):
    max, mean,  min = X.max(),X.mean(),X.min() 
    X_std = (X - mean) / (max - min)
    X_scaled = X_std * (max - min) + min
    return X_scaled
def collect_edge_weights(G, communities_dict):
    community_weights = {"community":[],"node":[],"neighbor":[],"weight":[]}
    community_node_weighted_sum = {"community":[],"node":[],"weight":[],"degree":[]} 
    for community, nodes in communities_dict.items():
        #if len(nodes) < 10:
        #    continue
        subGraph = nx.subgraph(G,nodes)
        for node in nodes:
            sum = []
            for neighbor in subGraph.neighbors(node):
                if subGraph.has_edge(node, neighbor):
                    community_weights["community"].append(community)
                    community_weights["node"].append(node)
                    community_weights["neighbor"].append(neighbor)
                    community_weights["weight"].append(subGraph[node][neighbor].get('weight', 1))
                    sum.append(community_weights["weight"][-1])
            community_node_weighted_sum["community"].append(community)
            community_node_weighted_sum["node"].append(node)
            community_node_weighted_sum["weight"].append(np.sum(sum))
            community_node_weighted_sum["degree"].append(len(sum))
    return pd.DataFrame.from_dict(community_weights),pd.DataFrame.from_dict(community_node_weighted_sum)
def within_module_z_score(G, communities):
    z_scores = {"node":[],"community":[],"z_scores":[]}
    for comm_id, nodes in communities.items():
        degrees = np.array([G.degree(node) for node in nodes])
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees) if len(degrees) > 1 else 1.0
        for node in nodes:
            z_scores["node"].append(node)
            z_scores["community"].append(comm_id)
            z_scores["z_scores"].append((G.degree(node) - mean_degree) / std_degree)
    return pd.DataFrame.from_dict(z_scores) 

def participation_coefficient(G, partition):
    """
    Compute the participation coefficient for each node in a graph.

    Parameters:
    - G: NetworkX Graph
    - partition: dict mapping each node to its community ID.

    Returns:
    - pd.DataFrame: A DataFrame containing the node, community, and participation coefficient.
    """
    part_coeff = {"node": [], "community": [], "part_coeff": []}
    total_degree = dict(G.degree())
    
    for node in G.nodes():
        community_id = partition[node]
        degree = total_degree[node]
        
        # Sum the edges between the node and neighbors in the same community
        comm_degrees = sum(1 for neigh in G.neighbors(node) if partition[neigh] == community_id)
        
        # Compute participation coefficient
        if degree > 0:
            score = 1 - (comm_degrees / degree) ** 2
        else:
            score = 0  # Isolated nodes have a participation coefficient of 0
        
        # Append to result
        part_coeff["node"].append(node)
        part_coeff["community"].append(community_id)
        part_coeff["part_coeff"].append(score)
    
    return pd.DataFrame.from_dict(part_coeff)

def compute_correlation_community(df, model, pruning_strategy):
    grouped = df.groupby('community')

    # Store the correlation results
    results = []

    # Iterate over each community group
    for community, group in grouped:
        z_scores = group['z_scores']
        part_coeff = group['part_coeff']
        
        # Calculate Pearson and Spearman correlation
        pearson_corr, _ = pearsonr(z_scores, part_coeff)
        spearman_corr, _ = spearmanr(z_scores, part_coeff)
        # Append results to a list
        results.append({
            'community': community,
            'community_size': group.shape[0],
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'model':model,
            'pruning_strategy':pruning_strategy 
        })
    return pd.DataFrame(results)



def get_modulesCommunityDataset(sparsity_ratio):
    with open("./dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_info_list = json.load(openfile)
    dataset_list = get_dataset_list(dataset_info_list)
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
    with open("result/dataMultidisciplinaryCognitiveSkillsFrameworkRestrict.json", 'r') as openfile:
        dataCategory = json.load(openfile)

    llama_distribution, _ = take_average(llama_distribution)
    vicuna_distribution, _ = take_average(vicuna_distribution)
    llama_chat_distribution, _ = take_average(llama_chat_distribution)
    
    distribution_dist = [llama_distribution,llama_chat_distribution,vicuna_distribution]
    original_dist = [llama_original,llama_chat_original,vicuna_original]    
    modules_community_dataset = create_projection_network(dataCategory,dataset_list, distribution_dist, original_dist, sparsity_ratio = sparsity_ratio,get_graph=True)
    return modules_community_dataset,dataset_info_list, dataset_list


if __name__ == "__main__":
    set_random_seed(int(2))
    sparsity_ratio = "20"
    modules=["attn.q", "attn.k", "attn.v", "attn.o","gate","mlp.up", "mlp.down"]
    all_modules = [f"{i}_{m}"  for i in range(3,31) for m in modules]
    model_path = "/gpfs/u/home/LLMG/LLMGbhnd/scratch/checkpoint/"
    plt.clf()
    plt.rcParams.update({
        'font.size': 16,           # Set the font size
        'text.usetex' : True,
        'font.family' : 'serif' ,
        'font.serif' : 'cm',
    })
    modules_community_dataset,dataset_info_list, dataset_list = get_modulesCommunityDataset(sparsity_ratio)
    for ps, model, network_prop in zip(modules_community_dataset["pruner_style"],modules_community_dataset["model"],modules_community_dataset["network_property"]):
        for c, cc in network_prop["community"].items():#modules_community_dataset["network_property"][0]["community"].items():
            print(ps,model,c)
            print("\t",sorted(cc),flush=True)


    data_zscore_part = []
    data_dist = []
    data_correlation = []
    for idx, model in enumerate(modules_community_dataset["model"]):
        pruner_style = modules_community_dataset["pruner_style"][idx]
        G = modules_community_dataset["graph"][idx]
        network_property = modules_community_dataset["network_property"][idx] 


        z_scores = within_module_z_score(G, network_property["community"])
        part_coeff = participation_coefficient(G, network_property["partition"])
        z_score_part_coeff = z_scores.merge(part_coeff, on=["node","community"], how="inner")
        z_score_part_coeff["pruner_style"] = [pruner_style]*z_score_part_coeff.shape[0] 
        z_score_part_coeff["model"] = [model]*z_score_part_coeff.shape[0] 
        
        community_weights,community_node_weighted_sum = collect_edge_weights(G, network_property["community"])
        community_weights["pruner_style"] = [pruner_style]*community_weights.shape[0] 
        community_weights["model"] = [model]*community_weights.shape[0] 
        
        data_zscore_part.append(z_score_part_coeff)
        data_dist.append(community_weights)
        data_correlation.append(compute_correlation_community(z_score_part_coeff, model, pruner_style))
        if idx%3 == 0:
            fig, ax = plt.subplots(figsize=(20,8),ncols=3, sharey=True)
            fig_node, ax_node = plt.subplots(figsize=(20,8),ncols=3,sharey=True)
            #fig.suptitle(f'Edge Weight Distribution | {pruner_style}')
            #fig_node.suptitle(f'Scatter Plot  | {pruner_style}') 
        print(community_weights)
        sns.histplot(data=community_weights, x="weight", hue="community", kde=True, bins=100, log_scale=True, palette="tab10",ax= ax[idx-floor(int(idx/3)*3)])
        ax[idx-floor(int(idx/3)*3)].set_title(r"$model$")
        ax[idx-floor(int(idx/3)*3)].set_xlabel(r'$Edge Weight$')
        ax[idx-floor(int(idx/3)*3)].set_ylabel(r'$Frequency$')
        h, l = ax[idx-floor(int(idx/3)*3)].get_legend_handles_labels()
        ax[idx-floor(int(idx/3)*3)].legend(h,l,loc="upper left",markerscale=3)
        
        sns.scatterplot(data=z_score_part_coeff,x="part_coeff",y="z_scores",hue="community",ax= ax_node[idx-floor(int(idx/3)*3)], palette="tab10")
        for comm,color in zip(z_score_part_coeff["community"].unique(),sns.color_palette("tab10", len(z_score_part_coeff["community"].unique()))):
            sns.regplot(data=z_score_part_coeff[z_score_part_coeff["community"]==comm],x="part_coeff",y="z_scores",ax= ax_node[idx-floor(int(idx/3)*3)], color=color,lowess=True)
        ax_node[idx-floor(int(idx/3)*3)].set_title(model)
        ax_node[idx-floor(int(idx/3)*3)].set_ylabel(r'$Z Score$')
        ax_node[idx-floor(int(idx/3)*3)].set_xlabel(r'$Participant Coefficient$')
        ax_node[idx-floor(int(idx/3)*3)].axhline(0)
        ax_node[idx-floor(int(idx/3)*3)].axvline(0)
        h, l = ax_node[idx-floor(int(idx/3)*3)].get_legend_handles_labels()
        ax_node[idx-floor(int(idx/3)*3)].legend(h,l,loc="upper left",markerscale=3)
        data_zscore_part.append(z_score_part_coeff)
        data_dist.append(community_weights)
        data_correlation.append(compute_correlation_community(z_score_part_coeff, model, pruner_style))
        
        '''sns.lineplot(data=community_node_weighted_sum, y="weight",x="degree", hue="community",  palette="tab10",ax= ax_node[pruner_idx])
        ax_node[pruner_idx].set_title(pruner_style)
        ax_node[pruner_idx].set_xlabel('Total Node Weight')
        ax_node[pruner_idx].set_ylabel('Frequency')'''
        if idx in  [2, 5]:
            fig.tight_layout()
            fig_node.tight_layout()
            fig.savefig(f'./figure/edge_distribution_{model}_{pruner_style}.pdf', dpi=360.0)
            fig_node.savefig(f'./figure/scatter_plot_{model}_{pruner_style}.pdf', dpi=360.0)

    data_zscore_part_concat = pd.concat(data_zscore_part, ignore_index=True)
    data_zscore_part_concat.to_csv("./result/z_scores_part_coefficient.csv", index=False)
    data_dist_concat = pd.concat(data_dist, ignore_index=True)
    data_dist_concat.to_csv("./result/distribution_comm.csv", index=False)
    data_corr_concat = pd.concat(data_correlation, ignore_index=True)
    data_corr_concat.to_csv("./result/z_scores_part_correlation.csv", index=False)
    print("DONE!") 