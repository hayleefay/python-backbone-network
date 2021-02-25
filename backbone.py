'''
This module implements the disparity filter to compute a significance score of edge weights in networks
'''

import networkx as nx
import numpy as np
from scipy import integrate
import pandas as pd
alpha = 0.05
DATA_PATH = "/Users/hayleeham/Documents/linkedin-network-viz/data/final/"
# which path are we doing?
network = "ll"
# paths
EDGES_IN = f"{network}/edges.csv"
NODES_IN = f"{network}/nodes.csv"
EDGES_OUT = f"{network}/backbone_edges.csv"
NODES_OUT = f"{network}/backbone_nodes.csv"


def build_network(nodes, edges):
    ''' Build the weighted network
        Args
            nodes: dataframe of node names and industries
            edges: dataframe of endpoints and weights of edges
        Returns
            Weighted graph object
    '''
    G = nx.Graph()
    # add the nodes
    G.add_nodes_from(nodes['gvkey'].values)
    # add the edges -- looking for (2, 3, {'weight': 3.1415})
    edge_values = edges.values
    edge_tuples = []
    if network == 'full':
        for source, target, weight, weight_norm in edge_values:
            edge_tuples.append((source, target, {'weight':weight}))
    else:
        for source, target, weight in edge_values:
            edge_tuples.append((source, target, {'weight':weight}))
    G.add_edges_from(edge_tuples)
    
    return G


def disparity_filter(G, weight='weight'):
    ''' Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009
        Args
            G: Weighted NetworkX graph
        Returns
            Weighted graph with a significance score (alpha) assigned to each edge
        References
            M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    '''
    B = nx.Graph()
    for u in G:
        k = len(G[u])
        if k > 1:
            sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
            for v in G[u]:
                w = G[u][v][weight]
                p_ij = float(np.absolute(w))/sum_w
                alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
                B.add_edge(u, v, weight = w, alpha=float('%.4f' % alpha_ij))
    return B
            

if __name__ == '__main__':
    # indicate which network working with
    print(NODES_IN.split('/')[0])
    # read in nodes and edges for the network
    nodes = pd.read_csv(DATA_PATH + NODES_IN)
    edges = pd.read_csv(DATA_PATH + EDGES_IN)
    print("Original", nodes.shape, edges.shape)

    # build the network
    print("Begin network build")
    G = build_network(nodes, edges)
    print("Network built")
    print('Network shape: nodes = %s, edges = %s' % (G.number_of_nodes(), G.number_of_edges()))

    # set alpha (constant) and run filter
    G = disparity_filter(G)
    print('Network shape: nodes = %s, edges = %s' % (G.number_of_nodes(), G.number_of_edges()))
    G2 = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d['alpha'] < alpha])
    print('alpha = %s' % alpha)
    print('original: nodes = %s, edges = %s' % (G.number_of_nodes(), G.number_of_edges()))
    print('backbone: nodes = %s, edges = %s' % (G2.number_of_nodes(), G2.number_of_edges()))

    # write out the edges
    nx.write_weighted_edgelist(G2, DATA_PATH + EDGES_OUT, delimiter=",")

    # read back in edges and add header
    edges_pruned = pd.read_csv(DATA_PATH + EDGES_OUT, names=["source", "target", "weight"])

    # filter out nodes with no edges
    source_nodes = list(edges_pruned['source'].values)
    target_nodes = list(edges_pruned['target'].values)
    nodes_pruned = nodes[nodes['gvkey'].isin(source_nodes + target_nodes)]
    print("Pruned nodes:", nodes_pruned.shape)
    print("Pruned edges:", edges_pruned.shape)

    # write out nodes and edges
    nodes_pruned.to_csv(DATA_PATH + NODES_OUT, index=False)
    edges_pruned.to_csv(DATA_PATH + EDGES_OUT, index=False)
    print(f"Nodes written out to {DATA_PATH + NODES_OUT}")
    print(f"Edges written out to {DATA_PATH + EDGES_OUT}")


