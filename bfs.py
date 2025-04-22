from mpi4py import MPI
import json
import argparse
import numpy as np
import time
from collections import deque
import numba
from numba import jit, prange
import sys

#[Console]::OutputEncoding = [System.Text.Encoding]::GetEncoding("windows-1251")


def load_graph_from_json(filename):
    """Завантажує граф з JSON файлу"""
    with open(filename, 'r') as f:
        graph_data = json.load(f)
    return graph_data


@jit(nopython=True, parallel=True)
def bfs_openmp(adjacency_list, start_node, num_nodes):
    """
    Реалізація BFS з використанням OpenMP (через Numba parallel)

    Args:
        adjacency_list: Список суміжності у форматі numba-сумісного типу даних
        start_node: Початкова вершина
        num_nodes: Кількість вершин

    Returns:
        distances: Масив відстаней від початкової вершини до всіх інших
    """
    distances = np.full(num_nodes, -1, dtype=np.int32)
    distances[start_node] = 0

    current_level = np.zeros(num_nodes, dtype=np.int32)
    next_level = np.zeros(num_nodes, dtype=np.int32)

    current_level[start_node] = 1
    current_distance = 1

    # Поки є вершини для обробки
    while np.sum(current_level) > 0:
        # Проходимо паралельно по всіх вершинах поточного рівня
        for node in prange(num_nodes):
            if current_level[node] == 1:
                # Для кожного сусіда
                for i in range(len(adjacency_list[node])):
                    neighbor = adjacency_list[node][i]
                    if distances[neighbor] == -1:
                        distances[neighbor] = current_distance
                        next_level[neighbor] = 1

        # Оновлюємо поточний рівень та збільшуємо відстань
        for i in prange(num_nodes):
            current_level[i] = next_level[i]
            next_level[i] = 0

        current_distance += 1

    return distances


def convert_adjacency_for_numba(adj_list_dict, num_nodes):
    """Конвертує словник списку суміжності в формат, зручний для Numba"""
    # Створюємо список списків (для Numba)
    max_neighbors = max(len(adj_list_dict[node]) for node in adj_list_dict)
    adj_list_numba = np.full((num_nodes, max_neighbors), -1, dtype=np.int32)

    for node in adj_list_dict:
        for i, (neighbor, _) in enumerate(adj_list_dict[node]):
            adj_list_numba[node][i] = neighbor

    return adj_list_numba


def bfs_sequential(adjacency_list, start_node, num_nodes):
    """
    Послідовна реалізація BFS для порівняння

    Args:
        adjacency_list: Список суміжності у вигляді словника
        start_node: Початкова вершина
        num_nodes: Кількість вершин

    Returns:
        distances: Словник відстаней від початкової вершини до всіх інших
    """
    distances = {node: -1 for node in range(num_nodes)}
    distances[start_node] = 0

    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        for neighbor, _ in adjacency_list[current]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


def bfs_mpi(adjacency_list, start_node, num_nodes):
    """
    Реалізація BFS з використанням MPI

    Args:
        adjacency_list: Список суміжності у вигляді словника
        start_node: Початкова вершина
        num_nodes: Кількість вершин

    Returns:
        distances: Словник відстаней від початкової вершини до всіх інших
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Розподілення вершин між процесами
    nodes_per_proc = num_nodes // size
    start_idx = rank * nodes_per_proc
    end_idx = start_idx + nodes_per_proc if rank < size - 1 else num_nodes

    # Ініціалізація відстаней
    local_distances = {node: -1 for node in range(num_nodes)}

    if rank == 0:
        # Початковий процес ініціалізує BFS
        local_distances[start_node] = 0
        current_level = [start_node]
    else:
        current_level = []

    level = 0

    while True:
        # Розсилаємо поточний рівень всім процесам
        current_level = comm.bcast(current_level, root=0)

        # Якщо нема вершин для обробки, завершуємо
        if not current_level:
            break

        level += 1
        local_next_level = []

        # Кожен процес обробляє свою частину вершин
        for node in current_level:
            neighbors = [n for n, _ in adjacency_list[node] if start_idx <= n < end_idx and local_distances[n] == -1]
            for neighbor in neighbors:
                local_distances[neighbor] = level
                local_next_level.append(neighbor)

        # Збираємо результати з усіх процесів
        all_next_levels = comm.gather(local_next_level, root=0)

        if rank == 0:
            # Об'єднуємо результати
            next_level = []
            for proc_level in all_next_levels:
                next_level.extend(proc_level)
            current_level = next_level

            # Оновлюємо глобальні відстані
            for proc_id in range(1, size):
                proc_distances = comm.recv(source=proc_id)
                for node, dist in proc_distances.items():
                    if dist != -1:
                        local_distances[node] = dist

            # Відправляємо оновлені відстані іншим процесам
            for proc_id in range(1, size):
                comm.send(local_distances, dest=proc_id)
        else:
            # Відправляємо локальні відстані процесу 0
            comm.send({n: d for n, d in local_distances.items() if d != -1 and start_idx <= n < end_idx}, dest=0)

            # Отримуємо оновлені глобальні відстані
            local_distances = comm.recv(source=0)

    # Синхронізуємо фінальні результати
    all_distances = comm.gather(local_distances, root=0)

    if rank == 0:
        final_distances = local_distances
        # Перевірка та об'єднання результатів
        for proc_distances in all_distances[1:]:
            for node, dist in proc_distances.items():
                if final_distances[node] == -1 and dist != -1:
                    final_distances[node] = dist
        return final_distances
    else:
        return None


def run_bfs_comparison(graph_data, start_node, use_mpi=False, use_openmp=False):
    """
    Запускає різні версії BFS і порівнює їх продуктивність

    Args:
        graph_data: Дані графа
        start_node: Початкова вершина
        use_mpi: Чи використовувати MPI
        use_openmp: Чи використовувати OpenMP

    Returns:
        results: Словник з результатами та часом виконання
    """
    adjacency_list = graph_data["adjacency_list"]
    num_nodes = graph_data["nodes"]

    # Конвертуємо рядкові ключі назад у цілі числа (якщо потрібно)
    if all(isinstance(k, str) for k in adjacency_list.keys()):
        adjacency_list = {int(k): v for k, v in adjacency_list.items()}

    results = {}

    # Послідовний BFS (завжди виконуємо для порівняння)
    start_time = time.time()
    seq_distances = bfs_sequential(adjacency_list, start_node, num_nodes)
    seq_time = time.time() - start_time
    results["sequential"] = {"time": seq_time, "distances": seq_distances}

    # MPI BFS
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        start_time = time.time()
        mpi_distances = bfs_mpi(adjacency_list, start_node, num_nodes)
        mpi_time = time.time() - start_time

        if rank == 0:
            results["mpi"] = {"time": mpi_time, "distances": mpi_distances}

    # OpenMP BFS
    if use_openmp:
        # Конвертуємо список суміжності для Numba
        adj_list_numba = convert_adjacency_for_numba(adjacency_list, num_nodes)

        start_time = time.time()
        openmp_distances = bfs_openmp(adj_list_numba, start_node, num_nodes)
        openmp_time = time.time() - start_time

        # Конвертуємо numpy масив у словник для порівняння
        openmp_distances_dict = {i: d for i, d in enumerate(openmp_distances)}
        results["openmp"] = {"time": openmp_time, "distances": openmp_distances_dict}

    # Верифікація результатів (опціонально)
    if "mpi" in results:
        mpi_correct = all(seq_distances[k] == results["mpi"]["distances"][k] for k in seq_distances)
        results["mpi"]["correct"] = mpi_correct

    if "openmp" in results:
        openmp_correct = all(seq_distances[k] == results["openmp"]["distances"][k] for k in seq_distances if
                             results["openmp"]["distances"][k] != -1)
        results["openmp"]["correct"] = openmp_correct

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BFS алгоритм (послідовний, MPI, OpenMP)')
    parser.add_argument('--input', type=str, default='graph.json', help='Вхідний JSON файл з графом')
    parser.add_argument('--start', type=int, default=0, help='Початкова вершина')
    parser.add_argument('--mpi', action='store_true', help='Використовувати MPI')
    parser.add_argument('--openmp', action='store_true', help='Використовувати OpenMP (через Numba)')

    args = parser.parse_args()

    # Завантаження графа
    graph_data = load_graph_from_json(args.input)

    # Запуск BFS
    results = run_bfs_comparison(graph_data, args.start, args.mpi, args.openmp)

    # Виведення результатів (тільки для головного процесу в MPI)
    if not args.mpi or MPI.COMM_WORLD.Get_rank() == 0:

        sys.stdout.reconfigure(encoding='utf-8')
        print("\nРезультати виконання BFS:")
        for method, data in results.items():
            print(f"{method.upper()}: {data['time']:.6f} секунд")
            if "correct" in data:
                print(f"  Правильність: {data['correct']}")

        # Обчислення прискорення
        if "mpi" in results:
            speedup = results["sequential"]["time"] / results["mpi"]["time"]
            print(f"Прискорення MPI: {speedup:.2f}x")

        if "openmp" in results:
            speedup = results["sequential"]["time"] / results["openmp"]["time"]
            print(f"Прискорення OpenMP: {speedup:.2f}x")