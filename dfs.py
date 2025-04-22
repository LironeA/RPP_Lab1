from mpi4py import MPI
import json
import argparse
import numpy as np
import time
from collections import deque
import numba
from numba import njit, prange
import traceback
import sys
import os


def load_graph_from_json(filename):
    """Завантажує граф з JSON файлу"""
    try:
        with open(filename, 'r') as f:
            graph_data = json.load(f)
        return graph_data
    except FileNotFoundError:
        print(f"Помилка: Файл '{filename}' не знайдено.")
        return None
    except json.JSONDecodeError:
        print(f"Помилка: Файл '{filename}' містить невалідний JSON.")
        return None
    except Exception as e:
        print(f"Помилка при завантаженні графа: {str(e)}")
        return None


@njit(parallel=True)
def dfs_openmp(adjacency_list, start_node, num_nodes):
    """
    Безпечна паралельна DFS (компонентний обхід) з Numba та OpenMP.
    Кожен потік обходить лише свою компоненту.
    """

    visited_global = np.zeros(num_nodes, dtype=np.uint8)  # глобальний масив для відвідуваних вершин
    order_global = np.full(num_nodes, -1, dtype=np.int32)  # глобальний масив для порядку вершин
    order_index = np.zeros(1, dtype=np.int32)  # для спільного порядку (atomic)

    for start in prange(num_nodes):  # Використовуємо prange замість range для паралельного виконання
        if visited_global[start] == 0:
            local_stack = np.empty(num_nodes, dtype=np.int32)
            local_top = 0
            local_stack[local_top] = start

            while local_top >= 0:
                node = local_stack[local_top]
                local_top -= 1

                if visited_global[node] == 0:
                    visited_global[node] = 1

                    # Атомарно оновлюємо order_global
                    idx = order_index[0]  # можна реалізувати атомарне оновлення вручну
                    order_global[idx] = node
                    order_index[0] += 1  # Оновлюємо лічильник

                    # Додаємо сусідів (в зворотному порядку для DFS)
                    for i in range(adjacency_list.shape[1] - 1, -1, -1):
                        neighbor = adjacency_list[node, i]
                        if neighbor != -1 and visited_global[neighbor] == 0:
                            local_top += 1
                            if local_top < num_nodes:  # перевірка переповнення стеку
                                local_stack[local_top] = neighbor

    return visited_global, order_global

def convert_adjacency_for_numba(adj_list_dict, num_nodes):
    """Конвертує словник списку суміжності в формат, зручний для Numba"""
    try:
        max_neighbors = max((len(adj_list_dict.get(node, [])) for node in range(num_nodes)), default=0)
        adj_list_numba = np.full((num_nodes, max_neighbors), -1, dtype=np.int32)

        for node in range(num_nodes):
            neighbors = adj_list_dict.get(node, [])
            for i, (neighbor, _) in enumerate(neighbors):
                if 0 <= neighbor < num_nodes:  # Перевірка валідності вершини
                    adj_list_numba[node][i] = neighbor
                else:
                    print(f"Попередження: Ігноруємо невалідну вершину {neighbor}")
                    adj_list_numba[node][i] = -1

        return adj_list_numba
    except Exception as e:
        print(f"Помилка при конвертації списку суміжності: {str(e)}")
        # Створюємо порожній масив як запасний варіант
        return np.full((num_nodes, 1), -1, dtype=np.int32)


def dfs_sequential(adjacency_list, start_node, num_nodes):
    """
    Послідовна реалізація DFS для порівняння

    Args:
        adjacency_list: Список суміжності у вигляді словника
        start_node: Початкова вершина
        num_nodes: Кількість вершин

    Returns:
        visited: Словник відвіданих вершин
        order: Порядок обходу вершин
    """
    try:
        visited = {node: False for node in range(num_nodes)}
        order = {node: -1 for node in range(num_nodes)}
        counter = [0]  # Використовуємо список для мутабельності

        def dfs_visit(node):
            try:
                visited[node] = True
                order[node] = counter[0]
                counter[0] += 1

                if node not in adjacency_list:
                    print(f"Попередження: Вершина {node} відсутня в списку суміжності")
                    return

                for neighbor, _ in adjacency_list[node]:
                    if neighbor not in visited:
                        print(f"Попередження: Вершина {neighbor} відсутня в словнику відвіданих вершин")
                        visited[neighbor] = False
                    if not visited[neighbor]:
                        dfs_visit(neighbor)
            except RecursionError:
                print(
                    f"Помилка: Досягнуто максимальну глибину рекурсії. Спробуйте використати нерекурсивну реалізацію.")
            except Exception as e:
                print(f"Помилка в dfs_visit для вершини {node}: {str(e)}")

        # Запускаємо з початкової вершини
        if 0 <= start_node < num_nodes:
            dfs_visit(start_node)
        else:
            print(f"Помилка: Початкова вершина {start_node} поза межами графа")

        # Обробляємо всі компоненти зв'язності (якщо граф не зв'язний)
        for node in range(num_nodes):
            if not visited[node]:
                dfs_visit(node)

        return visited, order
    except Exception as e:
        print(f"Критична помилка в dfs_sequential: {str(e)}")
        return {node: False for node in range(num_nodes)}, {node: -1 for node in range(num_nodes)}


def dfs_mpi(adjacency_list, start_node, num_nodes):
    """
    Реалізація DFS з використанням MPI

    Args:
        adjacency_list: Список суміжності у вигляді словника
        start_node: Початкова вершина
        num_nodes: Кількість вершин

    Returns:
        visited: Словник відвіданих вершин
        order: Порядок обходу вершин
    """
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Розподілення вершин між процесами
        nodes_per_proc = max(1, num_nodes // size)  # Запобігаємо поділу на нуль
        start_idx = rank * nodes_per_proc
        end_idx = start_idx + nodes_per_proc if rank < size - 1 else num_nodes
        local_nodes = list(range(start_idx, end_idx))

        # Ініціалізація
        global_visited = np.zeros(num_nodes, dtype=np.int32)
        global_order = np.full(num_nodes, -1, dtype=np.int32)

        # Початкова вершина належить одному з процесів
        if start_idx <= start_node < end_idx or rank == 0:
            # Локальний DFS для початкової вершини і доступних сусідів
            stack = [start_node]
            visited_nodes = set()
            order_counter = 0

            while stack:
                try:
                    node = stack.pop()
                    if node not in visited_nodes:
                        visited_nodes.add(node)
                        global_visited[node] = 1
                        global_order[node] = order_counter
                        order_counter += 1

                        # Перевірка на наявність вершини в списку суміжності
                        if node not in adjacency_list:
                            continue

                        # Додаємо сусідів в зворотньому порядку (для правильного порядку DFS)
                        neighbors = sorted([(n, w) for n, w in adjacency_list[node]], reverse=True)
                        for neighbor, _ in neighbors:
                            if neighbor not in visited_nodes:
                                # Якщо сусід належить цьому процесу, додаємо в стек
                                if start_idx <= neighbor < end_idx:
                                    stack.append(neighbor)
                                # Інакше відправляємо запит на обробку власнику вузла
                                else:
                                    owner_rank = min(neighbor // nodes_per_proc, size - 1)
                                    try:
                                        comm.send((neighbor, order_counter), dest=owner_rank, tag=1)
                                    except Exception as e:
                                        print(f"Процес {rank}: Помилка при відправці до {owner_rank}: {str(e)}")
                except Exception as e:
                    print(f"Процес {rank}: Помилка в локальному DFS: {str(e)}")
                    continue

        # Обробка запитів від інших процесів
        request_counter = 0
        max_requests = 1000  # Обмеження для запобігання нескінченного циклу
        while request_counter < max_requests:
            try:
                # Перевіряємо наявність повідомлень
                status = MPI.Status()
                flag = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

                if flag:
                    source = status.Get_source()
                    tag = status.Get_tag()

                    if tag == 1:  # Запит на обробку вузла
                        try:
                            node, current_order = comm.recv(source=source, tag=tag)

                            # Якщо вузол ще не відвіданий, обробляємо його
                            if global_visited[node] == 0:
                                stack = [node]
                                order_counter = current_order

                                while stack:
                                    current = stack.pop()
                                    if global_visited[current] == 0:
                                        global_visited[current] = 1
                                        global_order[current] = order_counter
                                        order_counter += 1

                                        # Перевірка на наявність вершини в списку суміжності
                                        if current not in adjacency_list:
                                            continue

                                        neighbors = sorted([(n, w) for n, w in adjacency_list[current]], reverse=True)
                                        for neighbor, _ in neighbors:
                                            if global_visited[neighbor] == 0:
                                                # Якщо сусід належить цьому процесу, додаємо в стек
                                                if start_idx <= neighbor < end_idx:
                                                    stack.append(neighbor)
                                                # Інакше відправляємо запит на обробку власнику вузла
                                                else:
                                                    owner_rank = min(neighbor // nodes_per_proc, size - 1)
                                                    try:
                                                        comm.send((neighbor, order_counter), dest=owner_rank, tag=1)
                                                    except Exception as e:
                                                        print(
                                                            f"Процес {rank}: Помилка при відправці до {owner_rank}: {str(e)}")

                            # Повідомляємо процес-відправник про завершення обробки
                            comm.send(True, dest=source, tag=2)
                        except Exception as e:
                            print(f"Процес {rank}: Помилка при обробці запиту: {str(e)}")
                            # Все одно відправляємо відповідь, щоб уникнути блокування
                            comm.send(True, dest=source, tag=2)

                    elif tag == 2:  # Підтвердження обробки
                        _ = comm.recv(source=source, tag=tag)

                    elif tag == 3:  # Сигнал завершення
                        _ = comm.recv(source=source, tag=tag)
                        break

                else:
                    # Якщо немає повідомлень і процес закінчив свою роботу
                    # Перевіряємо, чи всі процеси завершили обробку
                    if rank == 0:
                        all_done = True
                        for p in range(1, size):
                            comm.send(True, dest=p, tag=3)
                        break
                    request_counter += 1
                    time.sleep(0.01)  # Зменшуємо навантаження на процесор
            except Exception as e:
                print(f"Процес {rank}: Помилка в циклі обробки повідомлень: {str(e)}")
                request_counter += 1
                time.sleep(0.01)  # Даємо час на відновлення

        # Збираємо результати з усіх процесів
        try:
            all_visited = comm.gather(global_visited, root=0)
            all_orders = comm.gather(global_order, root=0)
        except Exception as e:
            print(f"Процес {rank}: Помилка при зборі результатів: {str(e)}")
            return None, None

        if rank == 0:
            try:
                # Об'єднуємо результати
                final_visited = np.zeros(num_nodes, dtype=np.int32)
                final_order = np.full(num_nodes, -1, dtype=np.int32)

                for visited_arr in all_visited:
                    for i in range(num_nodes):
                        if visited_arr[i] == 1:
                            final_visited[i] = 1

                for order_arr in all_orders:
                    for i in range(num_nodes):
                        if order_arr[i] != -1 and (final_order[i] == -1 or order_arr[i] < final_order[i]):
                            final_order[i] = order_arr[i]

                # Конвертуємо до словників для сумісності з іншими реалізаціями
                visited_dict = {i: bool(final_visited[i]) for i in range(num_nodes)}
                order_dict = {i: final_order[i] for i in range(num_nodes)}

                return visited_dict, order_dict
            except Exception as e:
                print(f"Процес {rank}: Помилка при об'єднанні результатів: {str(e)}")
                return None, None
        else:
            return None, None
    except Exception as e:
        print(f"Критична помилка в dfs_mpi: {str(e)}")
        if rank == 0:
            return {node: False for node in range(num_nodes)}, {node: -1 for node in range(num_nodes)}
        else:
            return None, None


def run_dfs_comparison(graph_data, start_node, use_mpi=False, use_openmp=False):
    """
    Запускає різні версії DFS і порівнює їх продуктивність

    Args:
        graph_data: Дані графа
        start_node: Початкова вершина
        use_mpi: Чи використовувати MPI
        use_openmp: Чи використовувати OpenMP

    Returns:
        results: Словник з результатами та часом виконання
    """
    try:
        if graph_data is None:
            print("Помилка: Дані графа відсутні")
            return {}

        adjacency_list = graph_data.get("adjacency_list")
        if adjacency_list is None:
            print("Помилка: Відсутній список суміжності в даних графа")
            return {}

        num_nodes = graph_data.get("nodes")
        if num_nodes is None:
            print("Помилка: Відсутня кількість вершин в даних графа")
            # Спробуємо визначити кількість вершин з adjacency_list
            num_nodes = max(int(k) for k in adjacency_list.keys()) + 1 if adjacency_list else 0
            print(f"Автоматично визначена кількість вершин: {num_nodes}")

        # Конвертуємо рядкові ключі назад у цілі числа (якщо потрібно)
        try:
            if all(isinstance(k, str) for k in adjacency_list.keys()):
                adjacency_list = {int(k): v for k, v in adjacency_list.items()}
        except Exception as e:
            print(f"Помилка при конвертації ключів: {str(e)}")
            return {}

        # Перевірка валідності початкової вершини
        if not 0 <= start_node < num_nodes:
            print(f"Помилка: Початкова вершина {start_node} поза межами графа (0-{num_nodes - 1})")
            start_node = 0  # Встановлюємо початкову вершину на 0 як запасний варіант

        results = {}

        # Послідовний DFS (завжди виконуємо для порівняння)
        try:
            start_time = time.time()
            seq_visited, seq_order = dfs_sequential(adjacency_list, start_node, num_nodes)
            seq_time = time.time() - start_time
            results["sequential"] = {"time": seq_time, "visited": seq_visited, "order": seq_order}
        except Exception as e:
            print(f"Критична помилка в послідовному DFS: {str(e)}")
            traceback.print_exc()
            results["sequential"] = {"time": 0, "error": str(e)}

        # MPI DFS
        if use_mpi:
            try:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()

                start_time = time.time()
                mpi_visited, mpi_order = dfs_mpi(adjacency_list, start_node, num_nodes)
                mpi_time = time.time() - start_time

                if rank == 0:
                    if mpi_visited is not None and mpi_order is not None:
                        results["mpi"] = {"time": mpi_time, "visited": mpi_visited, "order": mpi_order}
                    else:
                        results["mpi"] = {"time": mpi_time, "error": "Помилка при виконанні MPI DFS"}
            except Exception as e:
                print(f"Критична помилка в MPI DFS: {str(e)}")
                traceback.print_exc()
                results["mpi"] = {"time": 0, "error": str(e)}

        # OpenMP DFS
        if use_openmp:
            try:
                # Конвертуємо список суміжності для Numba
                adj_list_numba = convert_adjacency_for_numba(adjacency_list, num_nodes)
                os.environ["NUMBA_NUM_THREADS"] = "4"
                start_time = time.time()
                openmp_visited, openmp_order = dfs_openmp(adj_list_numba, start_node, num_nodes)
                openmp_time = time.time() - start_time

                # Конвертуємо numpy масиви у словники для порівняння
                openmp_visited_dict = {i: bool(v) for i, v in enumerate(openmp_visited)}
                openmp_order_dict = {i: o for i, o in enumerate(openmp_order)}
                results["openmp"] = {
                    "time": openmp_time,
                    "visited": openmp_visited_dict,
                    "order": openmp_order_dict
                }
            except Exception as e:
                print(f"Критична помилка в OpenMP DFS: {str(e)}")
                traceback.print_exc()
                results["openmp"] = {"time": 0, "error": str(e)}

        # Верифікація результатів (опціонально)
        if "sequential" in results and "visited" in results["sequential"]:
            try:
                seq_set = set(i for i, v in results["sequential"]["visited"].items() if v)

                if "mpi" in results and "visited" in results["mpi"]:
                    mpi_set = set(i for i, v in results["mpi"]["visited"].items() if v)
                    results["mpi"]["correct"] = (seq_set == mpi_set)
                    if not results["mpi"]["correct"]:
                        print(f"Попередження: Результати MPI відрізняються від послідовних.")
                        print(f"  - Послідовний: {len(seq_set)} вершин")
                        print(f"  - MPI: {len(mpi_set)} вершин")

                if "openmp" in results and "visited" in results["openmp"]:
                    openmp_set = set(i for i, v in results["openmp"]["visited"].items() if v)
                    results["openmp"]["correct"] = (seq_set == openmp_set)
                    if not results["openmp"]["correct"]:
                        print(f"Попередження: Результати OpenMP відрізняються від послідовних.")
                        print(f"  - Послідовний: {len(seq_set)} вершин")
                        print(f"  - OpenMP: {len(openmp_set)} вершин")
            except Exception as e:
                print(f"Помилка при верифікації результатів: {str(e)}")

        return results
    except Exception as e:
        print(f"Критична помилка в run_dfs_comparison: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='DFS алгоритм (послідовний, MPI, OpenMP)')
        parser.add_argument('--input', type=str, default='graph.json', help='Вхідний JSON файл з графом')
        parser.add_argument('--start', type=int, default=0, help='Початкова вершина')
        parser.add_argument('--mpi', action='store_true', help='Використовувати MPI')
        parser.add_argument('--openmp', action='store_true', help='Використовувати OpenMP (через Numba)')

        args = parser.parse_args()

        # Перевірка існування файлу
        if not os.path.isfile(args.input):
            print(f"Помилка: Файл '{args.input}' не існує.")
            sys.exit(1)

        # Завантаження графа
        graph_data = load_graph_from_json(args.input)
        if graph_data is None:
            print(f"Помилка при завантаженні графа з файлу '{args.input}'")
            sys.exit(1)

        # Запуск DFS
        results = run_dfs_comparison(graph_data, args.start, args.mpi, args.openmp)

        # Виведення результатів (тільки для головного процесу в MPI)
        if not args.mpi or MPI.COMM_WORLD.Get_rank() == 0:
            print("\nРезультати виконання DFS:")
            for method, data in results.items():
                if "error" in data:
                    print(f"{method.upper()}: Помилка - {data['error']}")
                else:
                    print(f"{method.upper()}: {data.get('time', 0):.6f} секунд")
                    if "correct" in data:
                        print(f"  Правильність: {data['correct']}")

            # Обчислення прискорення
            if "mpi" in results and "sequential" in results and "time" in results["sequential"] and "time" in results[
                "mpi"] and results["sequential"]["time"] > 0:
                speedup = results["sequential"]["time"] / results["mpi"]["time"]
                print(f"Прискорення MPI: {speedup:.2f}x")

            if "openmp" in results and "sequential" in results and "time" in results["sequential"] and "time" in \
                    results["openmp"] and results["sequential"]["time"] > 0:
                speedup = results["sequential"]["time"] / results["openmp"]["time"]
                print(f"Прискорення OpenMP: {speedup:.2f}x")
    except Exception as e:
        print(f"Критична помилка в головному блоці програми: {str(e)}")
        traceback.print_exc()
        sys.exit(1)