import json 
import math
import numpy as np
import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt
# Import necessary libraries
mpl.rcParams.update(mpl.rcParamsDefault)
import collections
from utils.bag_of_words.skill_dataset import create_plot_bog_skills
from utils.bag_of_words.bipartite_multipartite_projection import spectral_sparsification, create_plot_bog_modules, get_projection, create_biparitite 
from utils.bag_of_words.network_property import get_network_property
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



def all_skill_label():
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
    return cognitive_skills_community, all_skill_label




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
    BC_dataset_modules, module_label = create_plot_bog_modules(distribution,original, dataset_list,pruner_style=pruner_style, pruner_ratio=sparsity_ratio,norm="|W|_0",plot=False, alpha=alpha1)
    A_skill_modules =  np.dot(AB_dataset_skill.T,BC_dataset_modules)
    sparse_network = spectral_sparsification(A_skill_modules, alpha2)
    sparse_network =  sparse_network #min_max(sparse_network)
    if modules_vs_modules:
        _ ,A_modules_modules  = get_projection(sparse_network, plot_projection= False)
        G, network_property = get_network_property(A_modules_modules,module_label,module_label )
    else:
        A_skills_skills , _ = get_projection(sparse_network, plot_projection= False)
        G, network_property = get_network_property(A_skills_skills,skill_label,skill_label )
    return G, network_property, (sparse_network,skill_label,module_label), (AB_dataset_skill, BC_dataset_modules, A_skill_modules)

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

def check_kl_divergence(community_detected_frequency, dataset_frequency,top_k=10,epsilon=1e-8):
    distributions = np.array([freq for _, freq in community_detected_frequency.items()],dtype='float64')
    data = {comm:{"high": None, "low": None} for comm in community_detected_frequency}
    distributions /= distributions.sum(axis=1, keepdims=True)
    dataset_name = [d for d in dataset_frequency]
    for samp in dataset_frequency:
        sample = np.array(dataset_frequency[samp],dtype='float64')+ epsilon
        # Normalize the distributions and sample if not already normalized
        sample /= sample.sum()
        dataset_frequency[samp] = sample
    for comm, comm_skill_dist in zip(community_detected_frequency,distributions):
            # Compute Kullback-Leibler Divergence from the sample to each distribution
            comm_skill_dist = comm_skill_dist + epsilon
            comm_skill_dist /= comm_skill_dist.sum()

            #kl_divergences = [entropy(comm_skill_dist, dataset_skills_dist) for _, dataset_skills_dist in dataset_frequency.items()]
            kl_divergences = [entropy(dataset_skills_dist, comm_skill_dist) for _, dataset_skills_dist in dataset_frequency.items()]
            #Find the distribution with the minimum KL divergence
            sorted_idx = np.argsort(kl_divergences)

            data[comm]["all"] =  [dataset_name[i] for i in sorted_idx]
            data[comm]["high"] =  [dataset_name[i] for i in sorted_idx[0:top_k]]
            data[comm]["low"] =  [dataset_name[i] for i in sorted_idx[-top_k:]]
    return data

def get_datasets_shared_by_modules(community_detected ,dataset_modules,dataset_label,module_label):
    community = {}
    for comm in community_detected:
        community_modules_list = community_detected[comm]
        modules_indices = [module_label.index(m) for m in community_modules_list]

        contributions = np.sum(dataset_modules[:, modules_indices], axis=1)

        ranked_nodes = sorted(zip(dataset_label, contributions), key=lambda x: x[1], reverse=True)
        community[comm] = [x for x, y in ranked_nodes]
    return community

def create_projection_network(dataCategory,dataset_list, distribution_dist, original_dist, sparsity_ratio = "25", get_graph=False):
    cognitive_skills_community, all_label_skills = all_skill_label()
    ground_truth = get_ground_truth(all_label_skills,cognitive_skills_community)
    ground_truth = np.array([comm for  _, comm in ground_truth.items()])
    if get_graph:
        data  = {"pruner_style":[],"model":[],"sparsity_ratio":[],"community":{"kl":[],"network":[]},"network_data":[],"graph":[],"network_property":[],"frequency_skill":[]}
    else:
        data  = {"pruner_style":[],"model":[],"sparsity_ratio":[],"community":{"kl":[],"network":[]},"network_data":[],"network_property":[],"frequency_skill":[]}
    for pruner_style in ["block","channel"]:
        for model_idx ,model in enumerate(["llama","llama_chat","vicuna"]):
            G, network_property, (skills_modules,skill_label,module_label), ( skill_dataset, dataset_modules, _)= get_community_for_alpha(dataCategory, dataset_list, distribution_dist[model_idx], original_dist[model_idx], pruner_style=pruner_style, sparsity_ratio=sparsity_ratio,alpha1=0.01,alpha2=0.01)
            community_detected_frequency = get_skills_shared_by_modules(G,network_property["community"],skills_modules,skill_label, module_label, all_label_skills)
            dataset_frequency = create_frequency_skills(dataCategory, all_label_skills)
            kl_based_data = check_kl_divergence(community_detected_frequency,dataset_frequency, top_k=20)
            nework_based_data = get_datasets_shared_by_modules(network_property["community"] ,dataset_modules,dataset_list,module_label)
            data["pruner_style"].append(pruner_style)
            data["sparsity_ratio"].append(sparsity_ratio)
            data["model"].append(model)
            data["network_property"].append(network_property)
            data["network_data"].append(((skill_label,dataset_list,module_label),(skill_dataset, dataset_modules,skills_modules)))
            if get_graph:
                data["graph"].append(G)
            save_data_kl = {}
            save_data_network = {}
            for comm, communities in network_property["community"].items():  
                save_data_kl[comm] = {"modules":communities,"dataset": kl_based_data[comm]}
                save_data_network[comm] = {"modules":communities,"dataset": nework_based_data[comm]}
            data["community"]["kl"].append(save_data_kl)
            data["community"]["network"].append(save_data_network)
            data["frequency_skill"].append(community_detected_frequency)
    return data