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


def dijkstra_sequential(adjacency_list, start_node, num_nodes):
    dist = [float('inf')] * num_nodes
    dist[start_node] = 0
    visited = [False] * num_nodes
    heap = [(0, start_node)]

    while heap:
        current_dist, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True

        for v, weight in adjacency_list[u]:
            if not visited[v] and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(heap, (dist[v], v))

    return dist


def dijkstra_mpi(adjacency_list, start_node, num_nodes):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nodes_per_proc = num_nodes // size
    start_idx = rank * nodes_per_proc
    end_idx = start_idx + nodes_per_proc if rank < size - 1 else num_nodes

    dist = np.full(num_nodes, np.inf)
    if start_idx <= start_node < end_idx:
        dist[start_node] = 0.0
    visited = np.zeros(num_nodes, dtype=bool)

    local_changed = True

    while True:
        all_changed = comm.allreduce(local_changed, op=MPI.LOR)
        if not all_changed:
            break

        local_changed = False

        for u in range(start_idx, end_idx):
            if visited[u] or dist[u] == np.inf:
                continue
            visited[u] = True
            for v, weight in adjacency_list[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    local_changed = True

        comm.Allreduce(MPI.IN_PLACE, dist, op=MPI.MIN)

    final_dist = comm.gather(dist, root=0)
    if rank == 0:
        result = np.full(num_nodes, np.inf)
        for proc_dist in final_dist:
            result = np.minimum(result, proc_dist)
        return result.tolist()
    else:
        return None


@jit(nopython=True, parallel=True)
def dijkstra_openmp(adj_matrix, start_node):
    num_nodes = adj_matrix.shape[0]
    dist = np.full(num_nodes, np.inf, dtype=np.float64)
    visited = np.zeros(num_nodes, dtype=np.bool_)
    dist[start_node] = 0.0

    for _ in range(num_nodes):
        u = -1
        min_dist = np.inf
        for i in range(num_nodes):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
        if u == -1:
            break

        visited[u] = True

        for v in range(num_nodes):
            if not visited[v] and adj_matrix[u][v] > 0:
                if dist[u] + adj_matrix[u][v] < dist[v]:
                    dist[v] = dist[u] + adj_matrix[u][v]

    return dist


def convert_to_matrix(adjacency_list, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for u in adjacency_list:
        for v, w in adjacency_list[u]:
            matrix[u][v] = w
    return matrix


def run_dijkstra_comparison(graph_data, start_node, end_node=None, use_mpi=False, use_openmp=False):
    adjacency_list = graph_data["adjacency_list"]
    num_nodes = graph_data["nodes"]

    if all(isinstance(k, str) for k in adjacency_list.keys()):
        adjacency_list = {int(k): v for k, v in adjacency_list.items()}

    results = {}

    start_time = time.time()
    seq_dist = dijkstra_sequential(adjacency_list, start_node, num_nodes)
    seq_time = time.time() - start_time
    results["sequential"] = {"time": seq_time, "distances": seq_dist}

    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        start_time = time.time()
        mpi_dist = dijkstra_mpi(adjacency_list, start_node, num_nodes)
        mpi_time = time.time() - start_time
        if rank == 0:
            results["mpi"] = {"time": mpi_time, "distances": mpi_dist}

    if use_openmp:
        matrix = convert_to_matrix(adjacency_list, num_nodes)
        start_time = time.time()
        omp_dist = dijkstra_openmp(matrix, start_node)
        omp_time = time.time() - start_time
        results["openmp"] = {"time": omp_time, "distances": omp_dist.tolist()}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dijkstra алгоритм (послідовний, MPI, OpenMP)')
    parser.add_argument('--input', type=str, default='graph.json', help='Вхідний JSON файл з графом')
    parser.add_argument('--start', type=int, default=0, help='Початкова вершина')
    parser.add_argument('--end', type=int, help='Кінцева вершина (не використовується в алгоритмі, збережено для сумісності)')
    parser.add_argument('--mpi', action='store_true', help='Використовувати MPI')
    parser.add_argument('--openmp', action='store_true', help='Використовувати OpenMP (через Numba)')

    args = parser.parse_args()
    graph_data = load_graph_from_json(args.input)
    results = run_dijkstra_comparison(graph_data, args.start, args.end, args.mpi, args.openmp)

    if not args.mpi or MPI.COMM_WORLD.Get_rank() == 0:
        print("\nРезультати виконання Dijkstra:")
        for method, data in results.items():
            print(f"{method.upper()}: {data['time']:.6f} секунд")
        if "mpi" in results:
            speedup = results["sequential"]["time"] / results["mpi"]["time"]
            print(f"Прискорення MPI: {speedup:.2f}x")
        if "openmp" in results:
            speedup = results["sequential"]["time"] / results["openmp"]["time"]
            print(f"Прискорення OpenMP: {speedup:.2f}x")
