from mpi4py import MPI
import json
import argparse
import numpy as np
import time
import heapq
from numba import jit, prange

def load_graph_from_json(filename):
    with open(filename, 'r') as f:
        graph_data = json.load(f)
    return graph_data

# -------------------- SEQUENTIAL KRUSKAL --------------------
def kruskal_sequential(graph_data):
    parent = list(range(graph_data["nodes"]))

    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_v] = root_u
            return True
        return False

    edges = []
    for u in graph_data["adjacency_list"]:
        for v, w in graph_data["adjacency_list"][u]:
            if u < v:  # уникаємо дублювання ребер
                edges.append((w, u, v))

    edges.sort()
    mst = []
    total_weight = 0
    for w, u, v in edges:
        if union(u, v):
            mst.append((u, v, w))
            total_weight += w

    return mst, total_weight

# -------------------- MPI KRUSKAL --------------------
def kruskal_mpi(graph_data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    all_edges = []
    for u in graph_data["adjacency_list"]:
        for v, w in graph_data["adjacency_list"][u]:
            if u < v:
                all_edges.append((w, u, v))

    all_edges.sort()
    chunk_size = len(all_edges) // size
    local_edges = all_edges[rank * chunk_size:(rank + 1) * chunk_size if rank != size - 1 else len(all_edges)]

    parent = list(range(graph_data["nodes"]))

    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_v] = root_u
            return True
        return False

    local_mst = []
    local_weight = 0
    for w, u, v in local_edges:
        if union(u, v):
            local_mst.append((u, v, w))
            local_weight += w

    all_msts = comm.gather(local_mst, root=0)
    all_weights = comm.gather(local_weight, root=0)

    if rank == 0:
        combined_edges = [edge for mst in all_msts for edge in mst]
        combined_edges.sort()
        parent = list(range(graph_data["nodes"]))
        mst = []
        total_weight = 0
        for u, v, w in combined_edges:
            if union(u, v):
                mst.append((u, v, w))
                total_weight += w
        return mst, total_weight
    return None, None

# -------------------- OPENMP PRIM --------------------
@jit(nopython=True, parallel=True)
def prim_openmp(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    selected = np.zeros(num_nodes, dtype=np.bool_)
    key = np.full(num_nodes, np.inf)
    parent = np.full(num_nodes, -1)
    key[0] = 0

    for _ in range(num_nodes):
        u = -1
        min_key = np.inf
        for i in range(num_nodes):
            if not selected[i] and key[i] < min_key:
                min_key = key[i]
                u = i
        if u == -1:
            break

        selected[u] = True

        for v in range(num_nodes):
            if adj_matrix[u][v] > 0 and not selected[v] and adj_matrix[u][v] < key[v]:
                key[v] = adj_matrix[u][v]
                parent[v] = u

    mst = []
    total_weight = 0.0
    for v in range(1, num_nodes):
        if parent[v] != -1:
            mst.append((int(parent[v]), v, float(adj_matrix[parent[v]][v])))
            total_weight += adj_matrix[parent[v]][v]
    return mst, total_weight

def convert_to_matrix(adjacency_list, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for u in adjacency_list:
        for v, w in adjacency_list[u]:
            matrix[u][v] = w
            matrix[v][u] = w  # MST потребує неорієнтованого графа
    return matrix

# -------------------- COMPARISON FUNCTION --------------------
def run_mst_comparison(graph_data, use_mpi=False, use_openmp=False):
    adjacency_list = graph_data["adjacency_list"]
    num_nodes = graph_data["nodes"]

    if all(isinstance(k, str) for k in adjacency_list.keys()):
        adjacency_list = {int(k): v for k, v in adjacency_list.items()}
        graph_data["adjacency_list"] = adjacency_list

    results = {}

    start_time = time.time()
    mst_seq, weight_seq = kruskal_sequential(graph_data)
    seq_time = time.time() - start_time
    results["sequential"] = {"time": seq_time, "weight": weight_seq, "mst": mst_seq}

    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start_time = time.time()
        mst_mpi, weight_mpi = kruskal_mpi(graph_data)
        mpi_time = time.time() - start_time
        if rank == 0:
            results["mpi"] = {"time": mpi_time, "weight": weight_mpi, "mst": mst_mpi}

    if use_openmp:
        matrix = convert_to_matrix(adjacency_list, num_nodes)
        start_time = time.time()
        mst_omp, weight_omp = prim_openmp(matrix)
        omp_time = time.time() - start_time
        results["openmp"] = {"time": omp_time, "weight": weight_omp, "mst": mst_omp}

    return results
