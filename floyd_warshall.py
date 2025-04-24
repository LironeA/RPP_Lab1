from mpi4py import MPI
import json
import argparse
import numpy as np
import time
import numba
import ctypes
import os


def load_graph_from_json(filename):
    """Завантажує граф з JSON файлу"""
    with open(filename, 'r') as f:
        graph_data = json.load(f)
    return graph_data

path = os.path.abspath("floyd_warshall_omp.dll")
lib = ctypes.CDLL(path)

# Визначення типів аргументів
lib.floyd_warshall_omp.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

def floyd_warshall_openmp(adjacency_matrix, num_nodes):
    """
    Реалізація алгоритму Флойда-Варшалла з використанням OpenMP (через Numba parallel)

    Args:
        adjacency_matrix: Матриця суміжності з вагами
        num_nodes: Кількість вершин

    Returns:
        distances: Матриця найкоротших шляхів між усіма парами вершин
    """

    n = adjacency_matrix.shape[0]
    dist = adjacency_matrix.copy().astype(np.float64)
    ptr = dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.floyd_warshall_omp(ptr, n)
    return dist


def floyd_warshall_sequential(adjacency_matrix, num_nodes):
    """
    Послідовна реалізація алгоритму Флойда-Варшалла

    Args:
        adjacency_matrix: Матриця суміжності з вагами
        num_nodes: Кількість вершин

    Returns:
        distances: Матриця найкоротших шляхів між усіма парами вершин
    """
    # Працюємо з копією, щоб не змінювати вхідну матрицю
    distances = np.copy(adjacency_matrix)

    # Основний алгоритм Флойда-Варшалла
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distances[i][k] != np.inf and distances[k][j] != np.inf:
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

    return distances


def floyd_warshall_mpi(adjacency_matrix, num_nodes):
    """
    Реалізація алгоритму Флойда-Варшалла з використанням MPI

    Args:
        adjacency_matrix: Матриця суміжності з вагами
        num_nodes: Кількість вершин

    Returns:
        distances: Матриця найкоротших шляхів між усіма парами вершин
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Більш надійний розподіл рядків між процесами
    rows_per_proc = num_nodes // size
    extra = num_nodes % size

    # Визначаємо початковий та кінцевий рядки для кожного процесу
    if rank < extra:
        start_row = rank * (rows_per_proc + 1)
        end_row = start_row + rows_per_proc + 1
    else:
        start_row = rank * rows_per_proc + extra
        end_row = start_row + rows_per_proc

    # Кожен процес працює тільки зі своєю частиною матриці
    local_matrix = adjacency_matrix[start_row:end_row].copy()

    # Зберігаємо повну матрицю тільки для обміну k-ми рядками
    full_matrix = adjacency_matrix.copy() if rank == 0 else None

    # Основний цикл алгоритму Флойда-Варшалла
    for k in range(num_nodes):
        # Для кожної ітерації k, розсилаємо k-й рядок всім процесам
        k_row = comm.bcast(adjacency_matrix[k].copy() if rank == 0 else None, root=0)

        # Кожен процес оновлює свою частину матриці
        for i in range(end_row - start_row):
            for j in range(num_nodes):
                if local_matrix[i][k] != np.inf and k_row[j] != np.inf:
                    local_matrix[i][j] = min(local_matrix[i][j], local_matrix[i][k] + k_row[j])

        # Збираємо оновлені k-і рядки для наступної ітерації
        if k >= start_row and k < end_row:
            # Якщо k-ий рядок в локальній матриці цього процесу, відправляємо його
            row_to_send = local_matrix[k - start_row].copy()
        else:
            row_to_send = None

        # Визначаємо, який процес містить цей рядок
        owner = 0
        for p in range(size):
            p_start = p * rows_per_proc + min(p, extra)
            p_end = p_start + rows_per_proc + (1 if p < extra else 0)
            if p_start <= k < p_end:
                owner = p
                break

        # Збираємо оновлений k-ий рядок на головний процес
        updated_k_row = comm.gather(row_to_send if rank == owner else None, root=0)

        # Оновлюємо повну матрицю на головному процесі
        if rank == 0 and updated_k_row[owner] is not None:
            full_matrix[k] = updated_k_row[owner]
            adjacency_matrix = full_matrix.copy()  # оновлюємо для наступної ітерації

    # Збираємо всі результати на головний процес
    gathered_results = comm.gather((start_row, local_matrix), root=0)

    # Об'єднуємо результати
    if rank == 0:
        result = np.copy(adjacency_matrix)
        for proc_start, proc_matrix in gathered_results:
            result[proc_start:proc_start + proc_matrix.shape[0]] = proc_matrix
        return result
    else:
        return None


def run_floyd_warshall_comparison(graph_data, use_mpi=False, use_openmp=False, use_sequential=True):
    """
    Запускає різні версії алгоритму Флойда-Варшалла і порівнює їх продуктивність

    Args:
        graph_data: Дані графа
        use_mpi: Чи використовувати MPI
        use_openmp: Чи використовувати OpenMP
        use_sequential: Чи використовувати послідовний алгоритм

    Returns:
        results: Словник з результатами та часом виконання
    """
    adjacency_matrix = np.array(graph_data["adjacency_matrix"], dtype=float)
    num_nodes = graph_data["nodes"]

    results = {}

    # Послідовний алгоритм Флойда-Варшалла (виконуємо лише якщо вказано use_sequential=True)
    seq_distances = None
    if use_sequential:
        start_time = time.time()
        seq_distances = floyd_warshall_sequential(adjacency_matrix, num_nodes)
        seq_time = time.time() - start_time
        results["sequential"] = {"time": seq_time, "distances": seq_distances}

    # MPI версія
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        start_time = time.time()
        mpi_distances = floyd_warshall_mpi(adjacency_matrix, num_nodes)
        mpi_time = time.time() - start_time

        if rank == 0:
            results["mpi"] = {"time": mpi_time, "distances": mpi_distances}

            # Перевірка правильності результатів (порівнюємо з послідовним тільки якщо він був виконаний)
            if seq_distances is not None:
                mpi_correct = np.allclose(seq_distances, mpi_distances, rtol=1e-5, atol=1e-8, equal_nan=True)
                results["mpi"]["correct"] = mpi_correct

    # OpenMP версія
    if use_openmp:
        start_time = time.time()
        openmp_distances = floyd_warshall_openmp(adjacency_matrix, num_nodes)
        openmp_time = time.time() - start_time

        results["openmp"] = {"time": openmp_time, "distances": openmp_distances}

        # Перевірка правильності результатів (порівнюємо з послідовним тільки якщо він був виконаний)
        if seq_distances is not None:
            openmp_correct = np.allclose(seq_distances, openmp_distances, rtol=1e-5, atol=1e-8, equal_nan=True)
            results["openmp"]["correct"] = openmp_correct

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Алгоритм Флойда-Варшалла (послідовний, MPI, OpenMP)')
    parser.add_argument('--input', type=str, default='graph.json', help='Вхідний JSON файл з графом')
    parser.add_argument('--mpi', action='store_true', help='Використовувати MPI')
    parser.add_argument('--openmp', action='store_true', help='Використовувати OpenMP')
    parser.add_argument('--sequential', action='store_true', help='Використовувати послідовний алгоритм')
    parser.add_argument('--output', type=str, default=None, help='Файл для збереження результатів')

    args = parser.parse_args()

    # Завантаження графа
    graph_data = load_graph_from_json(args.input)

    # Запуск алгоритму Флойда-Варшалла з новим параметром use_sequential
    results = run_floyd_warshall_comparison(graph_data, args.mpi, args.openmp, args.sequential)

    # Виведення результатів (тільки для головного процесу в MPI)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("\nРезультати виконання алгоритму Флойда-Варшалла:")
        for method, data in results.items():
            print(f"{method.upper()}: {data['time']:.6f} секунд")
            if "correct" in data:
                print(f"  Правильність: {data['correct']}")

        # Обчислення прискорення якщо послідовний алгоритм був виконаний
        if "sequential" in results:
            if "mpi" in results:
                speedup = results["sequential"]["time"] / results["mpi"]["time"]
                print(f"Прискорення MPI: {speedup:.2f}x")

            if "openmp" in results:
                speedup = results["sequential"]["time"] / results["openmp"]["time"]
                print(f"Прискорення OpenMP: {speedup:.2f}x")

        # Збереження результатів
        if args.output:
            # Підготуємо результати для збереження в JSON
            # Конвертуємо numpy масиви в списки
            json_results = {}
            for method, data in results.items():
                json_results[method] = {
                    "time": data["time"]
                }
                if "correct" in data:
                    json_results[method]["correct"] = data["correct"]

            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Результати збережено у файл: {args.output}")