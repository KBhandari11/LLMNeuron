
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from utils.bag_of_words.sparsification import *
from utils.bag_of_words.dataset_modules import *

def normalize_projection(matrix):
    degree_vector = np.sum(matrix, axis=1)
    degree_matrix = np.outer(degree_vector, degree_vector) 
    normalized_projection = np.divide(matrix, degree_matrix, where=degree_matrix!=0)  # Avoid division by zero
    return normalized_projection

def get_projection(A_skill_modules, normalize_proj=False, plot_projection= False):
    A_skill_skill = A_skill_modules @ A_skill_modules.T
    A_modules_modules = A_skill_modules.T @ A_skill_modules
    if normalize_proj:
        A_modules_modules = normalize_projection(A_modules_modules)
        A_skill_skill = normalize_projection(A_skill_skill)
    if plot_projection:
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)#'Greys'#
        sns.heatmap(A_skill_skill, ax=axis[0], cmap=cmap, annot=False, cbar=False)
        axis[0].set_title("Skills vs Skills")
        sns.heatmap(A_modules_modules, ax=axis[1], cmap=cmap, annot=False, cbar=False)
        axis[1].set_title("Modules vs Modules")
        plt.tight_layout()
        plt.show()
        #draw_network(A_skill_skill, skill_label)
    return A_skill_skill, A_modules_modules

def create_biparitite(AB,nodeA, nodeB):
    G = nx.Graph()
    rows, cols = AB.shape
    for i in range(rows):
        G.add_node(nodeA[i], bipartite=0)
    for j in range(cols):
        G.add_node(nodeB[j], bipartite=1)
    for i in range(rows):
        for j in range(cols):
            if AB[i, j] > 0:
                G.add_edge(nodeA[i],nodeB[j],weight=AB[i, j] )
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def create_multiparitite(AB,BC,nodeA, nodeB,nodeC):
    G = nx.Graph()
    sizeA, sizeB1 = AB.shape
    sizeB2, sizeC = BC.shape

    for i in range(sizeA):
        G.add_node(nodeA[i], subset=0)
    for j in range(sizeB1):
        G.add_node(nodeB[j], subset=1)
    for k in range(sizeC):
        G.add_node(nodeC[k], subset=2)

    for i in range(sizeA):
        for j in range(sizeB1):
            if AB[i, j] > 0:
                G.add_edge(nodeA[i],nodeB[j],weight=AB[i, j] )
    for i in range(sizeB2):
        for j in range(sizeC):
            if BC[i, j] > 0:
                G.add_edge(nodeB[i],nodeC[j],weight=BC[i, j] )
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def networkx_draw_bipartite(AB,A_node, B_node,A_node_color,B_node_color, title=""):
    G = create_biparitite(AB, A_node, B_node)
    X = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
    plt.figure(figsize=(50,30))
    layout=nx.drawing.layout.bipartite_layout(G, X,aspect_ratio=0.75,scale=0.5,align="horizontal")
    for edge in G.edges(data="weight"):
        nx.draw_networkx_edges(G, layout, edgelist=[edge])

    #nx.draw_networkx_nodes(G, pos=layout)
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos=layout, nodelist=[n for n, d in G.nodes(data=True) if d["bipartite"] == 0], node_color=A_node_color, **options)#dataset
    nx.draw_networkx_nodes(G, pos=layout, nodelist=[n for n, d in G.nodes(data=True) if d["bipartite"] == 1], node_color=B_node_color, **options)#green
    plt.title(title)
    #nx.draw_networkx_labels(G,layout,font_size=5,font_weight="bold",horizontalalignment='left', verticalalignment='top')
    plt.show()
def networkx_draw_tripartite(AB,BC,A_node, B_node,C_node,A_node_color,B_node_color,C_node_color,title=""):
    G = create_multiparitite(AB,BC,A_node, B_node,C_node)
    #X = {n for n, d in G.nodes(data=True) if d["subset"] == 0}
    plt.figure(figsize=(80,30))
    layout=nx.drawing.layout.multipartite_layout(G,scale=0.5,align="horizontal")
    for edge in G.edges(data="weight"):
        nx.draw_networkx_edges(G, layout, edgelist=[edge])

    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos=layout, nodelist=[n for n, d in G.nodes(data=True) if d["subset"] == 0], node_color=A_node_color, **options)#skill
    nx.draw_networkx_nodes(G, pos=layout, nodelist=[n for n, d in G.nodes(data=True) if d["subset"] == 1], node_color=B_node_color, **options)#dataset
    nx.draw_networkx_nodes(G, pos=layout, nodelist=[n for n, d in G.nodes(data=True) if d["subset"] == 2], node_color=C_node_color, **options)#modules
    #nx.draw_networkx_labels(G,layout,font_weight="bold",horizontalalignment='left', verticalalignment='top')
    plt.title(title)
    plt.show()

def plot_AC_bipartite(AB_dataset_skill, skill_label,distribution,original, dataset_list,pruner_style="block", pruner_ratio="15",norm="|W|_0",alpha1=0,alpha2=90):
    BC_dataset_modules,  module_label = create_plot_bog_modules(distribution,original, dataset_list,pruner_style=pruner_style, pruner_ratio=pruner_ratio,norm=norm,alpha=alpha1,plot=False)
    AC_skill_modules = np.dot(AB_dataset_skill.T,BC_dataset_modules)
    AC_skill_modules = spectral_sparsification(AC_skill_modules,alpha=alpha2)
    networkx_draw_bipartite(AC_skill_modules,skill_label,module_label, A_node_color="tab:red",B_node_color="tab:green")

