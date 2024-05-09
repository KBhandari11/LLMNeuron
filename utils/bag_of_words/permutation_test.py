import numpy as np
import networkx as nx
from scipy import stats
from networkx.algorithms import bipartite

def calculate_metrics(network):
    # Placeholder for your actual metric calculation
    # For demonstration, let's return random metrics
    result = {}
    result["average_degree"] = sum(dict(network.degree()).values()) / len(network)
    result["average_cluster"] = nx.average_clustering(network)  # Average clustering coefficient
    result["density"] = nx.density(network)
    result["global_efficiency"]  = nx.global_efficiency(network)
    result["assortativity_coefficient"]   = nx.degree_assortativity_coefficient(network)
    return result

def permute_network_edges(network):
    # Randomly shuffle the edges in the network
    if bipartite.is_bipartite(network):
        X, Y = bipartite.sets(network)
        degX, degY = bipartite.degrees(network, X)
        G = bipartite.configuration_model([degx for _, degx in  degX], [degy for _, degy in  degY])
    else:
        degree_distribution = [network.degree(node) for node in network.nodes()]
        G = nx.configuration_model(degree_distribution)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.Graph(G)

def statistic(x, y, axis):
    #print(np.mean(x, axis=axis), np.mean(y, axis=axis), flush=True)
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def permutation_test(original_network_list):
    original_metrics = {"average_degree":[],"average_cluster":[],"density":[],"global_efficiency":[],"assortativity_coefficient":[] }
    original_metrics_avg_value = {"average_degree":0,"average_cluster":0,"density":0,"global_efficiency":0,"assortativity_coefficient":0 }
    permuted_metrics_avg_value = {"average_degree":0,"average_cluster":0,"density":0,"global_efficiency":0,"assortativity_coefficient":0 }
    permuted_metrics = {"average_degree":[],"average_cluster":[],"density":[],"global_efficiency":[],"assortativity_coefficient":[] }

    for original_network in original_network_list:
        shuffled_network = permute_network_edges(original_network)
        original_network_metrics = calculate_metrics(original_network)
        shuffled_network_metrics = calculate_metrics(shuffled_network)
        for metric in original_network_metrics:
            original_metrics_avg_value[metric]+= original_network_metrics[metric]/len(original_network_list)
            permuted_metrics_avg_value[metric]+= shuffled_network_metrics[metric]/len(original_network_list)
            original_metrics[metric].append(original_network_metrics[metric])
            permuted_metrics[metric].append(shuffled_network_metrics[metric])


    p_values = {}
    for (metric, original_values), (metric, permuted_values)  in zip(original_metrics.items(),permuted_metrics.items()):
        data = np.array([[o_n,c_n] for o_n,c_n in zip(original_values,permuted_values )])
        '''percentile = percentileofscore(values, original_value)
        p_value = min(percentile, 100 - percentile) / 100.0 * 2 # Two-tailed test
        #print(f"Metric: {metric} | Original: {original_value} | Random: {values} | Percentile: {percentile} |p=  {p_value}")
        p_values[metric] = p_value'''
        result = stats.permutation_test((data[:,0], data[:,1]),statistic=statistic, vectorized=True,
                       n_resamples=np.inf, alternative='two-sided',permutation_type='independent')
        p_values[metric] = (result.pvalue, result.statistic, original_metrics_avg_value[metric], shuffled_network_metrics[metric])
    return p_values

# Example usage:
# original_network = nx.bipartite.random_graph(100, 100, 0.1) # Placeholder for your actual network
# p_values = permutation_test(original_network)
# print(p_values)

# To adjust for multiple comparisons, you can use Bonferroni correction or FDR as needed
