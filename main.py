#!/usr/bin/env python3
import argparse
import sys
import json
import os
import time
import numpy as np
from datetime import datetime
from mpi4py import MPI


def convert_to_python_type(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


def process_algorithm(algorithm, input_path, start, end, use_sequential, use_mpi, use_openmp):
    if algorithm == 'bfs':
        from bfs import load_graph_from_json, run_bfs_comparison
        graph_data = load_graph_from_json(input_path)
        return run_bfs_comparison(graph_data, start, use_mpi, use_openmp, use_sequential)

    elif algorithm == 'dfs':
        from dfs import load_graph_from_json, run_dfs_comparison
        graph_data = load_graph_from_json(input_path)
        return run_dfs_comparison(graph_data, start, use_mpi, use_openmp, use_sequential)

    elif algorithm == 'dijkstra':
        from dijkstra import load_graph_from_json, run_dijkstra_comparison
        graph_data = load_graph_from_json(input_path)
        return run_dijkstra_comparison(graph_data, start, end, use_mpi, use_openmp, use_sequential)

    elif algorithm == 'floyd_warshall':
        from floyd_warshall import load_graph_from_json, run_floyd_warshall_comparison
        graph_data = load_graph_from_json(input_path)
        return run_floyd_warshall_comparison(graph_data, use_mpi, use_openmp, use_sequential)

    elif algorithm == 'mst':
        from mst import load_graph_from_json, run_mst_comparison
        graph_data = load_graph_from_json(input_path)
        return run_mst_comparison(graph_data, use_mpi, use_openmp, use_sequential)

    else:
        raise ValueError(f"Невідомий алгоритм: {algorithm}")

#[Console]::OutputEncoding = [System.Text.Encoding]::GetEncoding("windows-1251")
def main():
    parser = argparse.ArgumentParser(description='Паралельні алгоритми на графах')
    parser.add_argument('--algorithm', type=str, required=True,
                        help='Алгоритм(и) через кому: bfs,dfs,dijkstra,floyd_warshall,mst')
    parser.add_argument('--input', type=str, default='graph.json', help='Вхідний JSON файл або директорія з графами')
    parser.add_argument('--start', type=int, default=0, help='Початкова вершина (для BFS, DFS, Dijkstra)')
    parser.add_argument('--end', type=int, default=None, help='Кінцева вершина (для Dijkstra)')
    parser.add_argument('--sequential', action='store_true', help='Виконувати послідовну реалізацію')
    parser.add_argument('--mpi', action='store_true', help='Використовувати MPI')
    parser.add_argument('--openmp', action='store_true', help='Використовувати OpenMP')
    parser.add_argument('--output', type=str, default=None, help='Файл або директорія для результатів')
    parser.add_argument('--batch', action='store_true', help='Обробити всі файли у директорії')

    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Якщо жоден метод не вибрано, автоматично включаємо послідовний
    if not (args.sequential or args.mpi or args.openmp):
        if rank == 0:
            print("Попередження: Жоден метод не вибрано. Автоматично включаємо послідовний метод.")
        args.sequential = True

    algorithms = [alg.strip() for alg in args.algorithm.split(',')]

    if args.batch:
        input_dir = os.path.abspath(args.input)  # Получаем абсолютный путь
        if not os.path.isdir(input_dir):
            if rank == 0:
                print("Помилка: вказаний шлях не є папкою.")
            sys.exit(1)

        graph_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

        for filename in graph_files:
            full_path = os.path.join(args.input, filename)
            for alg in algorithms:
                if rank == 0:
                    print(f"\n[{datetime.now()}] Старт обробки: {filename}, алгоритм: {alg}")
                    print(f"Алгоритм: {alg} | Sequential: {args.sequential} | MPI: {args.mpi} | OpenMP: {args.openmp}")
                try:
                    results = process_algorithm(alg, full_path, args.start, args.end, args.sequential, args.mpi,
                                                args.openmp)
                    if rank == 0 and results:
                        for method, data in results.items():
                            if 'time' in data:
                                print(f"{method.upper()}: {data['time']:.6f} секунд")
                                if 'correct' in data:
                                    print(f"  Правильність: {data['correct']}")
                            elif 'error' in data:
                                print(f"{method.upper()}: Помилка - {data['error']}")

                        # Обчислення прискорення, якщо є послідовна реалізація для порівняння
                        if 'sequential' in results and 'time' in results['sequential']:
                            seq_time = results['sequential']['time']
                            if seq_time > 0:
                                if 'mpi' in results and 'time' in results['mpi']:
                                    speedup = seq_time / results['mpi']['time']
                                    print(f"Прискорення MPI: {speedup:.2f}x")
                                if 'openmp' in results and 'time' in results['openmp']:
                                    speedup = seq_time / results['openmp']['time']
                                    print(f"Прискорення OpenMP: {speedup:.2f}x")

                        if args.output:
                            os.makedirs(args.output, exist_ok=True)
                            output_filename = f"{os.path.splitext(filename)[0]}_{alg}_result.json"
                            output_path = os.path.join(args.output, output_filename)
                            with open(output_path, 'w') as f:
                                json.dump(results, f, indent=2, default=convert_to_python_type)
                            print(f"-> Результат збережено у {output_path}")
                    if rank == 0:
                        print(f"[{datetime.now()}] Завершено: {filename}, алгоритм: {alg}")
                except Exception as e:
                    if rank == 0:
                        print(f"[{datetime.now()}] Помилка у {filename} з алгоритмом {alg}: {e}")
                        import traceback
                        traceback.print_exc()
    else:
        for alg in algorithms:
            if rank == 0:
                print(f"\n[{datetime.now()}] Старт обробки одного файлу: {args.input}, алгоритм: {alg}")
                print(f"Алгоритм: {alg} | Sequential: {args.sequential} | MPI: {args.mpi} | OpenMP: {args.openmp}")
            try:
                results = process_algorithm(alg, args.input, args.start, args.end, args.sequential, args.mpi,
                                            args.openmp)
                if rank == 0 and results:
                    for method, data in results.items():
                        if 'time' in data:
                            print(f"{method.upper()}: {data['time']:.6f} секунд")
                            if 'correct' in data:
                                print(f"  Правильність: {data['correct']}")
                        elif 'error' in data:
                            print(f"{method.upper()}: Помилка - {data['error']}")

                    # Обчислення прискорення, якщо є послідовна реалізація для порівняння
                    if 'sequential' in results and 'time' in results['sequential']:
                        seq_time = results['sequential']['time']
                        if seq_time > 0:
                            if 'mpi' in results and 'time' in results['mpi']:
                                speedup = seq_time / results['mpi']['time']
                                print(f"Прискорення MPI: {speedup:.2f}x")
                            if 'openmp' in results and 'time' in results['openmp']:
                                speedup = seq_time / results['openmp']['time']
                                print(f"Прискорення OpenMP: {speedup:.2f}x")

                    if args.output:
                        output_name = f"{os.path.splitext(os.path.basename(args.input))[0]}_{alg}_result.json"
                        output_path = args.output if not os.path.isdir(args.output) else os.path.join(args.output,
                                                                                                      output_name)
                        with open(output_path, 'w') as f:
                            json.dump(results, f, indent=2, default=convert_to_python_type)
                        print(f"Результати збережено у: {output_path}")
                if rank == 0:
                    print(f"[{datetime.now()}] Завершено обробку одного файлу: {args.input}, алгоритм: {alg}")
            except Exception as e:
                if rank == 0:
                    print(f"[{datetime.now()}] Помилка при запуску {alg} на файлі {args.input}: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()