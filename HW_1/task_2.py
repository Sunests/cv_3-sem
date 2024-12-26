import cv2
import numpy as np

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Преобразует изображение в HSV и создает маску для красных препятствий."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = mask_red_1 | mask_red_2
    return mask_red

def find_contours(mask: np.ndarray) -> list:
    """Находит контуры красных препятствий."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def analyze_lanes(image: np.ndarray, contours: list, num_lanes: int) -> int:
    """Анализирует полосы на наличие препятствий."""
    _, width, _ = image.shape
    lane_width = width // num_lanes
    lanes_with_obstacles = [False] * num_lanes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        lane_index = x // lane_width
        if lane_index < num_lanes:
            lanes_with_obstacles[lane_index] = True
    for i in range(num_lanes):
        if not lanes_with_obstacles[i]:
            return i
    return -1

def find_road_number(image: np.ndarray) -> int:
    """Найти номер полосы, на которой нет препятствия в конце пути."""
    num_lanes = 5
    mask = preprocess_image(image)
    contours = find_contours(mask)
    lane_number = analyze_lanes(image, contours, num_lanes)
    return lane_number


