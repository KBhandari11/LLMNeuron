import numpy as np
import networkx as nx
from scipy import stats
from networkx.algorithms import bipartite
import random 

def calculate_metrics(network):
    # Placeholder for your actual metric calculation
    # For demonstration, let's return random metrics
    def get_largest_connected_component(G):
        if nx.is_connected(G):
            return G
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            return G.subgraph(largest_cc).copy()


    
    result = {}
    result["average_degree"] = sum(dict(network.degree()).values()) / len(network)
    result["average_cluster"] = nx.average_clustering(network)  # Average clustering coefficient
    result["density"] = nx.density(network)
    result["global_efficiency"]  = nx.global_efficiency(network)
    result["assortativity_coefficient"]   = nx.degree_assortativity_coefficient(network)
    network_lcc = get_largest_connected_component(network)
    result["diameter"]  = nx.diameter(network_lcc)
    result["transitivity"]  =  nx.transitivity(network_lcc)
    result["avg_shorted_path_length"]  = nx.average_shortest_path_length(network_lcc)
    result["avg_betweenness_centrality"]  = np.mean(list(nx.betweenness_centrality(network_lcc).values()))
    result["avg_eigen_centrality"]  = np.mean(list(nx.eigenvector_centrality(network_lcc).values()))
    eigenvalues = np.linalg.eigvals(nx.adjacency_matrix(network_lcc).todense())
    result["spectral_radius"]  = max(eigenvalues)
    
    return result

def add_edges_to_match_degree_sequence_bipartite(B, desired_degrees):
    """
    Add edges to a bipartite graph B to match the desired degree sequence without parallel edges.
    
    B: A bipartite graph generated from nx.bipartite.configuration_model.
    desired_degrees: A dictionary with nodes as keys and desired degrees as values.
    """
    # Get the set of nodes with degree deficit
    nodes_with_deficit = {node: desired_degrees[node] - B.degree[node] if B.degree[node] < desired_degrees[node] else 0 for node in B.nodes }
    
    # Create a list of all possible edges without considering existing edges
    top_nodes, bottom_nodes = nx.bipartite.sets(B)
    possible_edges = [(u, v) for u in top_nodes for v in bottom_nodes if not B.has_edge(u, v)]
    
    # Shuffle the list of possible edges to randomize the selection
    random.shuffle(possible_edges)
    
    # Add edges to the graph until the degree sequence is satisfied
    for u, v in possible_edges:
        if nodes_with_deficit[u] > 0 and nodes_with_deficit[v] > 0:
            B.add_edge(u, v)
            nodes_with_deficit[u] -= 1
            nodes_with_deficit[v] -= 1
            
        # Stop if all degree deficits are resolved
        if all(degree == 0 for degree in nodes_with_deficit.values()):
            break
    
    return B
def add_edges_to_match_degree_sequence(G, desired_degrees):
    """
    Add or remove edges to match the desired degree sequence without parallel edges.
    
    G: A graph (can be any type) generated from nx.configuration_model or any other method.
    desired_degrees: A dictionary with nodes as keys and desired degrees as values.
    """
    # Calculate the degree deficits
    nodes_with_deficit = {node: desired_degrees[node] - G.degree[node] for node in G.nodes}
    
    # Create a list of all possible edges that can be added or removed
    possible_edges_add = [(u, v) for u in G.nodes for v in G.nodes if u != v]

    
    # Shuffle the possible edges to randomize the selection
    random.shuffle(possible_edges_add)
    
    # Add edges to satisfy positive degree deficits
    for u, v in possible_edges_add:
        if nodes_with_deficit[u] != 0 or nodes_with_deficit[v] != 0:
            if nodes_with_deficit[u] > 0 and nodes_with_deficit[v] > 0 and not G.has_edge(u, v):
                #print((u, v),"added")
                G.add_edge(u, v)
                nodes_with_deficit[u] = desired_degrees[u] - G.degree[u]
                nodes_with_deficit[v] = desired_degrees[v] - G.degree[v]
            if nodes_with_deficit[u] < 0 and nodes_with_deficit[v] < 0 and G.has_edge(u, v):
                #print((u, v),"removed")
                G.remove_edge(u, v) 
                nodes_with_deficit[u] = desired_degrees[u] - G.degree[u]
                nodes_with_deficit[v] = desired_degrees[v] - G.degree[v]

        if all(degree == 0 for degree in nodes_with_deficit.values()):
            return G
    return G

def relabel_configuration_bipartite(original_network, B):
    top_nodes_original, bottom_nodes_original = nx.bipartite.sets(original_network)
    top_nodes_generated, bottom_nodes_generated = nx.bipartite.sets(B)

    # Create mapping from generated nodes to original nodes
    top_mapping = {generated_node: original_node for generated_node, original_node in zip(sorted(top_nodes_generated), sorted(top_nodes_original))}
    bottom_mapping = {generated_node: original_node for generated_node, original_node in zip(sorted(bottom_nodes_generated), sorted(bottom_nodes_original))}

    # Combine the mappings
    mapping = {**top_mapping, **bottom_mapping}

    # Relabel nodes in B
    B = nx.relabel_nodes(B, mapping)
    return B

def permute_network_edges(network):
    # Randomly shuffle the edges in the network
    if bipartite.is_bipartite(network):
        X, Y = bipartite.sets(network)
        
        degX, degY = bipartite.degrees(network, X)
        G = bipartite.configuration_model([degX[node_x] for node_x in  X], [degY[node_y] for node_y in  Y])
        G.remove_edges_from(nx.selfloop_edges(G))
        G = relabel_configuration_bipartite(network, G)
        G = add_edges_to_match_degree_sequence_bipartite(G, dict(network.degree()))
    else:
        degree_sequence = [network.degree(node) for node in network.nodes()]
        #G = configuration_model(degree_sequence)
        print(network, flush=True)
        G = nx.configuration_model(degree_sequence,create_using=nx.Graph())
        print("Here",G,end=", ", flush=True)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.relabel_nodes(G, {generated_node: original_node for generated_node, original_node in zip(sorted(G.nodes()), sorted(network.nodes()))})
        #G = add_edges_to_match_degree_sequence(G, dict(network.degree()))
    return nx.Graph(G)

def statistic(x, y, axis):
    #print(np.mean(x, axis=axis), np.mean(y, axis=axis), flush=True)
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def permutation_test(original_network_list):
    
    original_metrics = {"average_degree":[],"average_cluster":[],"density":[],"global_efficiency":[],"assortativity_coefficient":[], 'diameter':[], 'transitivity':[], 'avg_shorted_path_length':[], 'avg_betweenness_centrality':[], 'avg_eigen_centrality':[], 'spectral_radius':[] }
    permuted_metrics = {"average_degree":[],"average_cluster":[],"density":[],"global_efficiency":[],"assortativity_coefficient":[], 'diameter':[], 'transitivity':[], 'avg_shorted_path_length':[], 'avg_betweenness_centrality':[], 'avg_eigen_centrality':[], 'spectral_radius':[] }

    for original_network in original_network_list:
        shuffled_network = permute_network_edges(original_network)
        print(original_network,"|",shuffled_network,flush=True)
        original_network_metrics = calculate_metrics(original_network)
        shuffled_network_metrics = calculate_metrics(shuffled_network)
        for metric in original_network_metrics:
            original_metrics[metric].append(original_network_metrics[metric])
            permuted_metrics[metric].append(shuffled_network_metrics[metric])


    p_values = {}
    for (metric, original_values), (metric, permuted_values)  in zip(original_metrics.items(),permuted_metrics.items()):
        data = np.array([[o_n,c_n] for o_n,c_n in zip(original_values,permuted_values )])
        result = stats.permutation_test((data[:,0], data[:,1]),statistic=statistic, vectorized=True,
                       n_resamples=999999, alternative='two-sided',permutation_type='independent')#np.inf
        print("\t\tpermutation_testing: ",metric,"pvalue:",result.pvalue, "statistics:",result.statistic, flush=True)
        p_values[metric] = {"pvalue":result.pvalue, "statistics":result.statistic, "original":original_metrics[metric], "random":permuted_metrics[metric]}
    return p_values

# Example usage:
# original_network = nx.bipartite.random_graph(100, 100, 0.1) # Placeholder for your actual network
# p_values = permutation_test(original_network)
# print(p_values)

# To adjust for multiple comparisons, you can use Bonferroni correction or FDR as needed
