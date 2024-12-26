import numpy as np
from collections import deque

def find_start(image: np.ndarray) -> tuple:
    """Находит стартовую точку в лабиринте."""
    return (0, np.where((image[0] == [255, 255, 255]).all(axis=1))[0][0])

def is_valid_neighbor(image: np.ndarray, neighbor: tuple) -> bool:
    """Проверяет, является ли соседний узел допустимым."""
    row, col = neighbor
    return (0 <= row < image.shape[0] and
            0 <= col < image.shape[1] and
            (image[row, col] == [255, 255, 255]).all())

def find_path(image: np.ndarray, start: tuple) -> tuple:
    """Ищет путь в лабиринте используя BFS."""
    end_row = image.shape[0] - 1
    queue = deque([start])
    visited = set()
    visited.add(start)
    parent = {start: None}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.popleft()

        if current[0] == end_row:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            x, y = zip(*path)
            return (np.array(x), np.array(y))

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if is_valid_neighbor(image, neighbor) and neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current

    return None

def find_way_from_maze(image: np.ndarray) -> tuple:
    """Найти путь через лабиринт."""
    start = find_start(image)
    return find_path(image, start)

