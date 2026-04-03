"""
classical_solvers.py
"""

import heapq
import itertools
from problem_encoding import cost, branch, is_solution, decode_solution


#greedy approximation for the MVC problem
def greedy_mvc(graph):
    """
    selects iteratively the vertex with the highest degree and removes its edges until the graph has no more edges
    so in the MVC we take each time the most connected vertex

    returns a list of vertices in the vertex cover and its cost
    """
    G_copy = graph.copy()
    cover = []

    while G_copy.number_of_edges() > 0:
        #find the vertex with the highest degree
        node = max(G_copy.nodes(), key=lambda x: G_copy.degree(x))
        cover.append(node)
        
        #delete the node and its edges from the graph
        G_copy.remove_node(node)

    return sorted(cover), len(cover)

#brute force solver for the MVC
def brute_force_mvc(graph):
    """
    Generates all possible subsets of vertices in increasing order of size
    The first subset that covers all edges is guaranteed to be the minimum vertex cover
    """
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    def is_valid_cover(cover_set):
        #check if every edge has at least one endpoint in the cover set
        for u, v in edges:
            if u not in cover_set and v not in cover_set:
                return False
        return True

    #test all combinations of size r (from 0 to the number of nodes)
    for r in range(len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            cover_set = set(subset)
            if is_valid_cover(cover_set):
                #the first valid cover is the minimum
                return sorted(list(subset)), r

    return [], 0
