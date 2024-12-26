import cv2
import numpy as np


import cv2
import numpy as np

def get_rotation_matrix(point: tuple, angle: float) -> np.ndarray:
    """Вычисляет матрицу поворота."""
    return cv2.getRotationMatrix2D(point, angle, scale=1.0)

def calculate_new_dimensions(M: np.ndarray, h: int, w: int) -> tuple:
    """Вычисляет новые размеры изображения после поворота."""
    # Используем np.dot вместо @
    left_upper = np.dot(M, np.array([0, 0, 1]))
    right_upper = np.dot(M, np.array([0, h, 1]))
    left_lower = np.dot(M, np.array([w, 0, 1]))
    right_lower = np.dot(M, np.array([w, h, 1]))

    low_w = np.min([left_lower[0], left_upper[0], right_lower[0], right_upper[0]])
    high_w = np.max([left_lower[0], left_upper[0], right_lower[0], right_upper[0]])
    new_w = int(np.round(high_w - low_w))

    low_h = np.min([left_lower[1], left_upper[1], right_lower[1], right_upper[1]])
    high_h = np.max([left_lower[1], left_upper[1], right_lower[1], right_upper[1]])
    new_h = int(np.round(high_h - low_h))

    return (low_w, low_h), (new_w, new_h)

def adjust_rotation_matrix(M: np.ndarray, offset: tuple) -> np.ndarray:
    """Корректирует матрицу поворота с учетом смещения."""
    M[0, 2] -= offset[0]  # Доступ к элементам матрицы через индексы
    M[1, 2] -= offset[1]
    return M

def rotate_image(image: np.ndarray, M: np.ndarray, new_dimensions: tuple) -> np.ndarray:
    """Выполняет поворот изображения с помощью аффинного преобразования."""
    return cv2.warpAffine(image, M, new_dimensions)

def rotate(image: np.ndarray, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутое изображение
    """
    h, w, _ = image.shape
    M = get_rotation_matrix(point, angle)
    offset, new_dimensions = calculate_new_dimensions(M, h, w)
    M = adjust_rotation_matrix(M, offset)
    rotated_image = rotate_image(image, M, new_dimensions)
    return rotated_image