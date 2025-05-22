import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.setrecursionlimit(3000)

torch.set_default_tensor_type(torch.cuda.FloatTensor
                              if torch.cuda.is_available()
                              else torch.FloatTensor)


def find(vertex, parent):
    if parent[vertex] != vertex:
        parent[vertex] = find(parent[vertex], parent)
    return parent[vertex]


def union(u, v, parent):
    root_u = find(u, parent)
    root_v = find(v, parent)
    if root_u != root_v:
        parent[root_u] = root_v


def find_cluster_graph(lattice, edges):
    # lattice: (n, length)
    # clause_idx_k: (m, k)
    n, length = lattice.shape
    if length < 2:
        return lattice, torch.zeros(1)
    label = torch.zeros_like(lattice, dtype=torch.int)
    label = label.reshape(-1)
    label[lattice.reshape(-1) > 0] = torch.arange(1, (lattice.reshape(-1) > 0).sum() + 1, dtype=torch.int)
    label = label.reshape(n, length)

    # edges = []
    # for clause_idx_k in clause_idx:
    #     k_sat = clause_idx_k.shape[1]
    #     for i in range(k_sat):
    #         for j in range(i + 1, k_sat):
    #             edges.append(torch.stack([clause_idx_k[:, i], clause_idx_k[:, j]], dim=1))
    # edges = torch.cat(edges, dim=0)
    # edges = torch.unique(edges, dim=0)
    equivalence = []
    for i in range(length):
        equivalence.append(label[:, i][edges])
    for i in range(length - 1):
        equivalence.append(torch.stack([label[:, i][edges[:, 0]], label[:, i + 1][edges[:, 1]]], dim=1))
        equivalence.append(torch.stack([label[:, i][edges[:, 1]], label[:, i + 1][edges[:, 0]]], dim=1))
    equivalence = torch.cat(equivalence, dim=0)

    nonzero_mask = (equivalence > 0).all(dim=1)
    equivalence = equivalence[nonzero_mask]
    equivalence = torch.unique(equivalence, dim=0)

    # find connected components of the equivalence graph
    graph_edges = equivalence.cpu().numpy()
    nodes = np.arange(1, lattice.sum().cpu().numpy().item() + 1)
    parent = {key: value for key, value in zip(nodes, nodes)}

    for edge in graph_edges:
        union(edge[0], edge[1], parent)

    value_map = torch.tensor([find(node, parent) for node in nodes])
    unique_labels = torch.unique(value_map)
    if unique_labels.numel() == 0:
        return label, torch.zeros(1)
    else:
        relabeled = torch.arange(1, len(unique_labels) + 1, dtype=torch.int)
        relabel_map = torch.zeros(unique_labels.max() + 1, dtype=torch.int)
        relabel_map[unique_labels] = relabeled
        value_map = relabel_map[value_map]

        value_map = torch.cat([torch.zeros(1, dtype=torch.int), value_map], dim=0)

        # relabel the lattice
        label = value_map[label]

        # Optional: each site can only be counted once within each cluster, remove the duplicates
        new_label, indices = label.sort(dim=1)
        new_label[:, 1:] *= (torch.diff(new_label, dim=1) != 0).to(torch.int64)
        indices = indices.sort(dim=1)[1]
        new_label = torch.gather(new_label, 1, indices)

        index = new_label.reshape(-1).to(torch.int64)
        weight = lattice.reshape(-1)
        cluster_sizes = torch.zeros(value_map.max() + 1, dtype=torch.float)
        cluster_sizes.scatter_add_(0, index, weight.to(torch.float))
        cluster_sizes = cluster_sizes[1:]

        return label, cluster_sizes
