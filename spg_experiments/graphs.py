import networkx as nx
import numpy as np
import torch


def build_fc_graph(elements, features=None):
    if not elements:
        graph = nx.complete_graph(3)
        nx.set_node_attributes(graph, features, "x")
        return graph

    graph = nx.Graph()

    for i, e0 in enumerate(elements):
        features = torch.cat([v for k, v in e0.items()])
        graph.add_node(i, x=to_numpy(features))
        for j, _ in enumerate(elements):
            if i != j:
                graph.add_edge(i, j)

    return graph


def to_numpy(input_arr):
    if isinstance(input_arr, torch.Tensor):
        return input_arr.cpu().detach().numpy()

    if isinstance(input_arr, np.ndarray):
        return input_arr

    if isinstance(input_arr, list):
        return np.array(input_arr)

    raise ValueError(f"Cannot convert {type(input_arr)} to Numpy")
