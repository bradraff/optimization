"""
Brad Rafferty
brjrafferty@gmail.com

This script attempts to find a Bayesian network structure that best fit some given data, where the fitness of the
structure is measured by the Bayesian score (defined in Decision Making Under Uncertainty, Section 2.4.1,
Mykel Kochenderfer). The script takes in .csv-formatted graphs and outputs the structure that best describe the
relationships between the nodes. The script implements the K2 search algorithm to achieve this functionality.
"""

import sys
import networkx as nx
import os
import math
import random
import itertools
import time
import matplotlib.pyplot as plt
from pandas import read_csv


def write_gph(dag, var_names, filename):
    """
    Writes a .gph file given the Directed Acyclic Graph (dag), variable names (var_names), and the output filename
    """
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(var_names[edge[0]], var_names[edge[1]]))


def check_cycles(graph_to_check):
    """
    Determines whether the input graph is acyclic or not
    """
    is_acyclic = nx.is_directed_acyclic_graph(graph_to_check)
    return is_acyclic


def initialize_graph(nodes_from_data):
    """
    Initializes a directed graph
    """
    # Produce new random seed to prevent repeating same shuffled order of nodes
    random.seed()
    # Initial graph structure, unshuffled (order of nodes presented in original .csv)
    nodes_shuffled = nodes_from_data.copy()
    # Initial graph structure, now shuffled
    random.shuffle(nodes_shuffled)
    graph_initialized = nx.DiGraph()
    graph_initialized.add_nodes_from(nodes_shuffled)
    return graph_initialized


def update_graph(graph_start, parent_to_add, node_of_interest):
    """
    loops through the parents_to_add graph without needing to know the names of keys or values
    """
    # Initialize the updated graph to be the starting graph
    graph_updated = graph_start
    graph_updated.add_edge(parent_to_add, node_of_interest)
    return graph_updated


def get_parent_instantiations(graph, data, node, **kwargs):
    """
    Returns all instantiations of the parent nodes in the graph for the given node
    """
    sample = kwargs.get('sample', 'empty')
    # Extract parents from networkx digraph (named predecessors by networkx)
    parents_of_node = list(graph.predecessors(node))
    values_of_parents = ()  # Tuple of possible values that the parents of the node can take on
    j = []  # We have to return something
    if len(parents_of_node) == 0:  # If this node has no parents
        qi = 1
        j = 0
    else:
        parent_insts = []  # Dict of possible instantiations of the parents; keys = parents, values = list of possible values the parent can take on
        for idx in range(len(parents_of_node)):
            parent = parents_of_node[idx]
            max_val_parent = max(data.iloc[:, parent].values)  # Max value that the parent can take on--will help determine range for instantiations
            # Add the list of possible parent values to the list parent_insts
            parent_insts.append(list(range(1, max_val_parent + 1)))
            if sample != 'empty':  # If a sample is input into the function, the user expects to calculate parent values as well
                # Extract actual parent value from data, add to tuple format
                values_of_parents += (data.iloc[sample, parent],)

        pi = list(itertools.product(*parent_insts))  # pi is the list of the various permutations (stored as tuples) of the parents of node i, i.e. parent instantiations
        qi = len(pi)
        if sample != 'empty':  # If a sample is input into the function, the user expects to calculate parent values as well
            # Convert the parent value combination to an index based on the ordering in pi
            j = get_idx_of_tuple(pi, values_of_parents)

    # Get ri:
    max_val_node = max(data.iloc[:, node].values)
    ri = max_val_node  # In these specific datasets, r_i is equal to the max value that X_i can take on, since X_i is discretized [1, 2,..., max_value]
    return j, qi, ri


def get_idx_of_tuple(input_list, value):
    """
    Returns the position of the value in a list, or raises a value error if the value is not in the list
    """
    for pos, t in enumerate(input_list):
        if t == value:
            return pos
    raise ValueError("list.index(c): x not in list")


def get_counts(graph, data):
    """
    get_counts determines the pseudo count alpha and its sum as well as the data count m and its sum for an input graph
    and the corresponding dataset
    :param graph: networkx digraph
    :param data: pandas dataframe
    :return: a, m,
    """
    [num_samples, _] = data.shape  # Number of [samples, nodes] in the data (each row is a sample, each column is a node)
    a = {}      # alpha, pseudo-count
    m = {}      # m, count
    nodes = list(graph.nodes)

    for sample in range(num_samples):
        #print('Sample {}'.format(sample))
        for idx in range(len(nodes)):  # for each node in graph
            node = nodes[idx]
            #print('\tNode {}'.format(node))
            i = node
            k = data.iloc[sample, node]  # value of node i in sample
            [j, _, _] = get_parent_instantiations(graph, data, node, sample=sample)
            ijk = "{}_{}_{}".format(i, j, k)
            ij0 = "{}_{}-{}".format(i, j, 0)
            if ij0 not in list(m.keys()): # If this ij0 has not been counted before
                m[ij0] = 0  # Initialize this m_ij0
                a[ij0] = 0  # Initialize
            if ijk not in list(m.keys()):  # If this ijk has not been counted before
                m[ijk] = 0  # Initialize this m_ijk
                a[ijk] = 1  # alpha_ijk per uniform prior
            a[ij0] += 1  # Count a_ij0
            m[ijk] += 1
            m[ij0] += 1
    return a, m


def calc_bayes_score(graph, data):
    """
    calc_bayes_score takes in a directed acyclic graph (graph) and a discretized dataset (data) and outputs the bayesian
    score of the graph (bayes_score)
    :param graph: networkx digraph structure
    :param data: pandas dataframe
    :return: bayes_score
    """
    bayes_score = 0  # Initialize the score
    [a, m] = get_counts(graph, data)
    nodes = list(graph.nodes)
    for i in range((len(nodes))):
        node = nodes[i]
        [_, qi, ri] = get_parent_instantiations(graph, data, node)
        for j in range(qi):
            ij0 = "{}_{}-{}".format(i,j,0)
            if ij0 in m.keys():
                bayes_score += math.lgamma(a[ij0]) - math.lgamma(a[ij0] + m[ij0])
            for k in range(ri):
                ijk = "{}_{}_{}".format(i,j,k)
                if ijk in m.keys():
                    bayes_score += math.lgamma(a[ijk] + m[ijk]) - math.lgamma(a[ijk])

    return bayes_score

def K2_search_algorithm(g_old, data, max_num_parents):
    """
    This algorithm heuristically searches for the most probably belief-network structure given a databae of cases
    """
    parents = {}
    score_old = calc_bayes_score(g_old, data)  # Bayes score for the initialized graph
    nodes = list(g_old.nodes)
    for ii in range(len(nodes)):
        node_of_interest = nodes[ii]
        parents[node_of_interest] = []
        parents_to_consider = nodes.copy()
        parents_to_consider.remove(node_of_interest)  # Can't use ith node as its own parent!
        # While there are still other nodes to consider as parents for current node and the current node has not exceeded max parents:
        while (len(parents_to_consider) > 0) and (len(parents[node_of_interest]) < max_num_parents):
            parent_to_consider = parents_to_consider[0]
            g_new = g_old
            g_new.add_edge(parent_to_consider, node_of_interest)

            score_new = calc_bayes_score(g_new, data)
            # If our new score is worse, reject this parent:
            if score_new <= score_old:
                g_new = g_old
                parents_to_consider.pop(0)
                score_new = score_old
                continue

            score_old = score_new
            g_old = g_new
            parents[node_of_interest].append(parent_to_consider)
            parents_to_consider.pop(0)

    return g_new, score_new


def compute(infile, outfile):
    data = read_csv(infile)
    [_, num_nodes] = data.shape
    max_num_parents = round(num_nodes / 3)  # Heuristic for max number of parents for a given node
    # Initialize a graph of random node ordering
    G_0 = initialize_graph(list(range(num_nodes)))
    # Perform K2 search algorithm to determine output graph G
    [G, B_score]  = K2_search_algorithm(G_0, data, max_num_parents)
    return G, B_score

# Compute the graph best on Bayesian scores
input_filename = "example.csv"
output_filename = "example.gph"
start = time.time()
[G, bayesian_score] = compute(input_filename, output_filename)
end = time.time()

# Print the results
print('The Bayes Score is {}'.format(bayesian_score))
print('The time taken was {} seconds'.format(end-start))

# Plot the results as a graph
nx.draw(G, with_labels=True)
plt.show()

# Write the output graph to the current directory
data = read_csv(input_filename)
variable_names = data.columns.to_list()
write_gph(G, variable_names, output_filename)
