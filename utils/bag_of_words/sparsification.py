import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import random


def sparse_matrix(matrix,alpha):
    flattened_matrix= np.sort(matrix.flatten())
    alpha_value= np.percentile(flattened_matrix,alpha)
    matrix[matrix < alpha_value] = 0
    return matrix

def bipartite_laplacian(adj_matrix):
    # Calculate degrees for nodes in U and V
    degrees_u = np.sum(adj_matrix, axis=1)
    degrees_v = np.sum(adj_matrix, axis=0)
    
    # Create diagonal degree matrices for U and V
    D_u = np.diag(degrees_u)
    D_v = np.diag(degrees_v)
    
    # Construct the bipartite Laplacian
    # Note: This construction depends on the specific definition and use case
    # For a simple case, we might construct a block matrix as follows:
    L = sp.bmat([[D_u, -adj_matrix], [-adj_matrix.T, D_v]], format='csr')
    return L

def weighted_laplacian(adj_matrix):
    """
    Calculate the Laplacian of a weighted graph represented by the adjacency matrix.
    """
    degrees = np.sum(adj_matrix, axis=0)
    D = np.diag(degrees)
    L = D - adj_matrix
    return L

def compute_effective_resistance(L):
    """
    Compute the effective resistance of each edge using the graph's Laplacian.
    L: Laplacian matrix of the graph.
    """
    epsilon = 1e-5
    L = L + epsilon * np.eye(L.shape[0])
    L_pseudo_inv = scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(L)).toarray()
    n = L.shape[0]
    R_eff = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            R_eff[i, j] = L_pseudo_inv[i, i] - 2 * L_pseudo_inv[i, j] + L_pseudo_inv[j, j]
            R_eff[j, i] = R_eff[i, j]
    return R_eff

def spectral_sparsification(adj_matrix, alpha=None):
    """
    Perform spectral sparsification on a graph represented by a weighted adjacency matrix.
    adj_matrix: Weighted adjacency matrix of the graph.
    alpha: Sampling probability multiplier.
    """
    # Calculate Laplacian and effective resistances
    if alpha == None:
        return adj_matrix

    original_m, original_n = adj_matrix.shape
    L = bipartite_laplacian(adj_matrix)
    adj_matrix = sp.bmat([[None, adj_matrix], [adj_matrix.T, None]], format='csr').toarray()

    R_eff = compute_effective_resistance(L)
    
    m, n = adj_matrix.shape
    sparsified_adj_matrix = np.zeros_like(adj_matrix)
    
    for i in range(m):
        for j in range(n):  # Assume undirected graph
            if adj_matrix[i, j] != 0:  # There's an edge
                weight = adj_matrix[i, j]
                p = min(1, alpha * R_eff[i, j] * (np.sum(adj_matrix[i]) + np.sum(adj_matrix[j])))#(np.log(n)/(epsilon**2)))#
                if np.random.rand() <= p:
                    # Keep the edge with probability p
                    sparsified_adj_matrix[i, j] = weight/p
                    sparsified_adj_matrix[j, i] = weight/p  # Since the graph is undirected
    sparsified_adj_matrix = sparsified_adj_matrix[:original_m, original_m:]
    sparsified_adj_matrix[sparsified_adj_matrix<=float(5e-5)] = 0
    return sparsified_adj_matrix