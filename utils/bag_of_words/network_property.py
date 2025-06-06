from community import community_louvain
import networkx as nx
from copy import deepcopy
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.bag_of_words.community_layout import community_layout
from sklearn.metrics import adjusted_rand_score, pairwise_distances
import numpy as np
from operator import itemgetter
from itertools import combinations

from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster,ClusterWarning, dendrogram
from scipy.spatial.distance import squareform
from warnings import simplefilter
simplefilter("ignore", ClusterWarning) 
def create_coassignment_matrix(node_list, all_runs_communities, num_runs):
    num_nodes = len(node_list)
    coassignment_matrix = np.zeros((num_nodes, num_nodes))

    node_index = {node: idx for idx, node in enumerate(node_list)}

    for run_communities in all_runs_communities:
        for community in run_communities.values():
            for i in range(len(community)):
                for j in range(i + 1, len(community)):
                    node_i = community[i]
                    node_j = community[j]
                    idx_i = node_index[node_i]
                    idx_j = node_index[node_j]

                    coassignment_matrix[idx_i, idx_j] += 1
                    coassignment_matrix[idx_j, idx_i] += 1  # Symmetric matrix

    coassignment_matrix /= num_runs

    return coassignment_matrix, node_index


def analyze_stable_communities(node_list,all_runs_communities,avg_num_communities, num_runs=100, threshold=0.8):


    coassignment_matrix, node_index = create_coassignment_matrix(node_list, all_runs_communities, num_runs)
    condensed_coassignment_matrix = coassignment_matrix# squareform(coassignment_matrix)
    Z = linkage(condensed_coassignment_matrix, method='average')
    clusters = fcluster(Z, t=Z[-avg_num_communities, 2], criterion='distance')
    '''# Create the heatmap
    # Plot dendrogram and heatmap together
    fig, ax_dendro = plt.subplots(figsize=(8, 4))

    # Create a subplot for the dendrogram
    dendrogram(Z, ax=ax_dendro, orientation='top', color_threshold=Z[-avg_num_communities, 2], labels=list(node_index.keys()))
    ax_dendro.set_yticks([])
    ax_dendro.set_title("Dendrogram")
    plt.show()
    fig,  ax_heatmap  = plt.subplots(figsize=(8, 8))
    # Create a subplot for the heatmap
    sns.heatmap(
        coassignment_matrix,
        ax=ax_heatmap,
        cmap='viridis',
        cbar=True,
        square=True,
        xticklabels=list(node_index.keys()),
        yticklabels=list(node_index.keys())
    )
    ax_heatmap.set_title("Co-Assignment Matrix Heatmap")

    # Show the plot
    plt.show()'''
    clustered_communities = defaultdict(list)
    for node, cluster_id in zip(node_list, clusters):
        clustered_communities[cluster_id].append(node)

    return dict(clustered_communities)

def plot_community_model(g,partition):
    g.remove_edges_from(nx.selfloop_edges(g))
    pos = community_layout(g, partition)
    d = dict(g.degree)
    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())
    modules=["attn.q", "attn.k", "attn.v", "attn.o","gate","mlp.up", "mlp.down"]
    resize_model_dict = {layer: [partition[f"{layer}_{m}"] if f"{layer}_{m}" in partition.keys() else max(partition.values())+1 for m in modules ] for layer in list(range(3,31))}
    #resize_model_dict = {layer: [partition[f"{layer}_{m}"] if f"{layer}_{m}" in partition.keys() else -1 for m in modules ] for layer in list(range(3,31))}
    df = pd.DataFrame(resize_model_dict,
                    index=modules)
    value_to_int = {value: i for i, value in enumerate(sorted(pd.unique(df.values.ravel())))}
    f, [ax1,ax2] = plt.subplots(figsize=(20,10), ncols=2)
    hm = sns.heatmap(df.replace(value_to_int).T, cmap="Accent", ax=ax1, cbar=False)
    # add legend
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    legend_ax = f.add_axes([box.x0 +box.width * 0.95, 0.5, 0, .1])
    #legend_ax = f.add_axes([.7, .5, 0, .1])
    legend_ax.axis('off')
    # reconstruct color map
    colors = plt.cm.Accent(np.linspace(0, 1, len(value_to_int)))
    # add color map to legend
    patches = [mpatches.Patch(facecolor=c, edgecolor=c) for c in colors]
    legend = legend_ax.legend(patches,
        sorted(value_to_int.keys()),
        handlelength=0.8, loc='lower left')
    for t in legend.get_texts():
        t.set_ha("left")
    community_color = [colors[partition[nodes]] for nodes in g.nodes()]
    nx.draw(g, pos, ax= ax2, node_color=community_color,nodelist=list(d.keys()),node_size=[v * 50 for v in d.values()],with_labels = True, edgelist=edges, edge_color=weights, width=5.0, edge_cmap=plt.cm.Blues)
    ax2.set_title(str(g))
    plt.show()

def compute_local_efficiency(graph):
    """
    Computes the local efficiency of each node in the graph.

    Parameters:
    - graph: A NetworkX graph.

    Returns:
    - A dictionary of nodes with their local efficiency.
    """
    efficiency = {}
    for node in graph.nodes():
        subgraph = graph.subgraph(graph.neighbors(node))
        if len(subgraph.nodes()) > 1:
            efficiency[node] = nx.global_efficiency(subgraph)
        else:
            efficiency[node] = 0
    return efficiency

def get_centrality(g, topNode=20):
    measures = {
        'degree_centrality': nx.degree_centrality(g),
        'betweenness_centrality': nx.betweenness_centrality(g),
        'closeness_centrality': nx.closeness_centrality(g),
        'eigenvector_centrality': nx.eigenvector_centrality(g, max_iter=1000),
        'clustering_coefficient': nx.clustering(g),
        'local_efficiency': compute_local_efficiency(g)
    }
    
    for measure, values in measures.items():
        sorted_nodes = sorted(values, key=lambda x: values[x], reverse=True)
        measures[measure] = sorted_nodes[:topNode]
    return measures
    
def plot_community(g,partition):
    plt.figure(figsize=(30,30))
    g.remove_edges_from(nx.selfloop_edges(g))
    pos = community_layout(g, partition)
    d = dict(g.degree)
    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())
    color = matplotlib.colormaps['Accent'].colors
    nx.draw(g, pos, node_color=[color[int(partition[nodes])] for nodes in g.nodes()],nodelist=list(d.keys()),node_size=[v * 50 for v in d.values()],with_labels = True, edgelist=edges, edge_color=weights, width=5.0, edge_cmap=plt.cm.Blues)
    plt.title(str(g),fontsize = 40)
    plt.show()
def heaviest(G):
    u, v, w = max(G.edges(data="weight"), key=itemgetter(2))
    return (u, v)

def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G, weight="weight")
    return max(centrality, key=centrality.get)


def get_community(g,community_seed=0,ground_truth=None):
    #g = deepcopy(G)
    #isolated = list(nx.isolates(g))
    #g.remove_nodes_from(isolated)
    #partition = community_louvain.best_partition(g)
    #communities = consensus_community(g, ground_truth =ground_truth)
    all_runs_communities = []
    num_communities_list = []
    num_runs = 100
    for i in range(num_runs):
        communities = nx.community.louvain_communities(g,resolution=1,weight="weight", seed=i)
        communities = {idx:list(x) for idx, x in enumerate(communities)}
        all_runs_communities.append(communities)
        num_communities_list.append(len(communities))  # Store number of communities per run
    avg_num_communities = int(np.mean(num_communities_list))
    communities = analyze_stable_communities(list(g.nodes()),all_runs_communities,avg_num_communities, num_runs=num_runs)
    communities = [set(value) for _, value in communities.items()]
    partition = {}
    for node in g.nodes():
        for idx,comm_nodes in enumerate(communities):
            if  node in comm_nodes:
                partition[node] = idx
                break
    comm = {x:[] for x in set(partition.values())}
    for i in partition:
        comm[partition[i]].append(i)
    '''if isolated != []:
        comm[len(set(partition.values()))+1] = isolated
        new  = set(partition.values())+1
        for node in isolated:
            partition[node] = new'''
    return comm, partition

def get_network_property(AB, labelA, labelB,community_seed=0,get_centrality_bool =False, ground_truth=None):
    G = nx.Graph()
    for i in range(AB.shape[0]):
        for j in range(AB.shape[1]):  
            if AB[i, j] > 0:
                G.add_edge(labelA[i], labelB[j], weight=AB[i, j])
    result = {}
    result["average_degree"] = sum(dict(G.degree()).values()) / len(G)
    result["average_cluster"] = nx.average_clustering(G)  # Average clustering coefficient
    result["density"] = nx.density(G)
    community , result["partition"]  = get_community(G,community_seed=community_seed)
    result["num_community"] = len([comm for comm in community if len(community[comm])!=0] )
    result["modularity"] = nx.community.modularity(G,[community[comm] for comm in community])
    result["community"] = community
    result["global_efficiency"]  = nx.global_efficiency(G)
    result["assortativity_coefficient"]   = nx.degree_assortativity_coefficient(G)
    if nx.is_connected(G):
        result["diameter"]  = nx.diameter(G)
        result["average_path_length"] = nx.average_shortest_path_length(G)
    else:
        g = deepcopy(G)
        S = g.subgraph(max(nx.connected_components(G), key=len))
        result["average_path_length"] = nx.average_shortest_path_length(S)
        result["diameter"]  = nx.diameter(S)

    if get_centrality_bool:
        result["centrality"]= get_centrality(G, topNode=20)
    return G, result