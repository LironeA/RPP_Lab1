import json
import random
import argparse
import numpy as np


def generate_graph(num_nodes, edge_probability=0.3, min_weight=1, max_weight=100, directed=False):
    """
    Генерує випадковий граф з вагами ребер.

    Args:
        num_nodes: Кількість вершин
        edge_probability: Ймовірність утворення ребра між вершинами
        min_weight: Мінімальна вага ребра
        max_weight: Максимальна вага ребра
        directed: Чи є граф орієнтованим

    Returns:
        Словник, що представляє граф у форматі: {
            "nodes": кількість вершин,
            "directed": чи орієнтований граф,
            "adjacency_list": список суміжності з вагами,
            "adjacency_matrix": матриця суміжності з вагами
        }
    """
    adjacency_list = {i: [] for i in range(num_nodes)}
    adjacency_matrix = np.inf * np.ones((num_nodes, num_nodes))
    np.fill_diagonal(adjacency_matrix, 0)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < edge_probability:
                weight = random.randint(min_weight, max_weight)
                adjacency_list[i].append((j, weight))
                adjacency_matrix[i][j] = weight

                if not directed:
                    if j not in [edge[0] for edge in adjacency_list[i]]:
                        adjacency_list[j].append((i, weight))
                        adjacency_matrix[j][i] = weight

    # Конвертуємо масив NumPy у список для JSON
    adjacency_matrix_list = adjacency_matrix.tolist()

    graph = {
        "nodes": num_nodes,
        "directed": directed,
        "adjacency_list": adjacency_list,
        "adjacency_matrix": adjacency_matrix_list
    }

    return graph


def save_graph_to_json(graph, filename):
    """Зберігає граф у JSON файл"""
    with open(filename, 'w') as f:
        json.dump(graph, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генератор випадкового графа')
    parser.add_argument('--nodes', type=int, default=100, help='Кількість вершин')
    parser.add_argument('--prob', type=float, default=0.3, help='Ймовірність ребра')
    parser.add_argument('--min_weight', type=int, default=1, help='Мінімальна вага ребра')
    parser.add_argument('--max_weight', type=int, default=100, help='Максимальна вага ребра')
    parser.add_argument('--directed', action='store_true', help='Чи граф орієнтований')
    parser.add_argument('--output', type=str, default='graph.json', help='Ім\'я вихідного файлу')

    args = parser.parse_args()

    graph = generate_graph(
        args.nodes,
        args.prob,
        args.min_weight,
        args.max_weight,
        args.directed
    )

    save_graph_to_json(graph, args.output)
    print(f"Граф збережено у файл: {args.output}")