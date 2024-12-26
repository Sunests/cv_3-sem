import numpy as np
import torch


def zero_pad(image_data: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
    """Applies zero-padding to the image with different horizontal and vertical padding."""
    return np.pad(image_data, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')


def get_padded_image(image_data: np.ndarray, kernel_height: int, kernel_width: int) -> np.ndarray:
    """Applies zero-padding to the image."""
    return np.pad(image_data, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)), mode='constant')


def perform_convolution(padded_image: np.ndarray, kernel_matrix: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    """Performs the convolution operation."""
    kernel_height, kernel_width = kernel_matrix.shape
    output = np.zeros((img_height, img_width))
    for row in range(img_height):
        for col in range(img_width):
            output[row, col] = np.sum(padded_image[row:row + kernel_height, col:col + kernel_width] * kernel_matrix)
    return output


def conv_nested(image_data: np.ndarray, kernel_matrix: np.ndarray) -> np.ndarray:
    """Conducts image convolution using a nested loop."""
    img_height, img_width = image_data.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output = np.zeros((img_height, img_width))

    padded_image = get_padded_image(image_data, kernel_height, kernel_width)
    flipped_kernel = np.flip(kernel_matrix)

    return perform_convolution(padded_image, flipped_kernel, img_height, img_width)


def conv_fast(image_data: np.ndarray, kernel_matrix: np.ndarray) -> np.ndarray:
    """Conducts fast image convolution using NumPy."""
    img_height, img_width = image_data.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output = np.zeros((img_height, img_width))

    padded_image = get_padded_image(image_data, kernel_height, kernel_width)
    flipped_kernel = np.flip(kernel_matrix)

    return perform_convolution(padded_image, flipped_kernel, img_height, img_width)


def conv_faster(image_data: np.ndarray, kernel_matrix: np.ndarray) -> np.ndarray:
    """Conducts image convolution using PyTorch for acceleration."""
    img_height, img_width = image_data.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output = np.zeros((img_height, img_width))

    flipped_kernel = np.flip(kernel_matrix)
    torch_kernel = torch.tensor(flipped_kernel.copy(), dtype=torch.float32).reshape(1, 1, kernel_height, kernel_width)

    conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_height, kernel_width), padding='same', bias=False)
    conv_layer.weight = torch.nn.Parameter(torch_kernel)

    torch_image = torch.tensor(image_data, dtype=torch.float32).reshape(1, 1, img_height, img_width)
    output = conv_layer(torch_image).detach().numpy().reshape((img_height, img_width))

    return output


def calculate_cross_correlation_coefficient(img_slice: np.ndarray, img_b_float: np.ndarray) -> float:
    """Calculates the cross-correlation coefficient."""
    coefficient = np.sqrt(np.sum(img_b_float ** 2) * np.sum(img_slice ** 2))
    if coefficient == 0:
        return 0  # Handle division by zero
    return np.sum(img_slice * img_b_float) / coefficient


def perform_cross_correlation(padded_image: np.ndarray, img_b_float: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    """Performs the cross-correlation operation."""
    kernel_height, kernel_width = img_b_float.shape
    output = np.zeros((img_height, img_width))
    for row in range(img_height):
        for col in range(img_width):
            img_slice = padded_image[row:row + kernel_height, col:col + kernel_width]
            output[row, col] = calculate_cross_correlation_coefficient(img_slice, img_b_float)
    return output


def cross_correlation(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Performs cross-correlation between two images."""
    img_b_float = image_b.astype(np.float64)
    img_a_float = image_a.astype(np.float64)

    img_height, img_width = img_a_float.shape
    kernel_height, kernel_width = img_b_float.shape

    padded_image = get_padded_image(img_a_float, kernel_height, kernel_width)

    return perform_cross_correlation(padded_image, img_b_float, img_height, img_width)



def zero_mean_cross_correlation(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Performs zero-mean cross-correlation."""
    temp_kernel = image_b - np.mean(image_b)
    return cross_correlation(image_a, temp_kernel)


def normalized_cross_correlation(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Performs normalized cross-correlation."""
    img_b_float = image_b.astype(np.float64)
    img_a_float = image_a.astype(np.float64)

    img_height, img_width = img_a_float.shape
    kernel_height, kernel_width = img_b_float.shape

    padded_image = get_padded_image(img_a_float, kernel_height, kernel_width)

    std_dev = np.std(img_b_float)
    mean_val = np.mean(img_b_float)
    norm_img_b = (img_b_float - mean_val) / std_dev if std_dev !=0 else img_b_float #handle division by zero

    return perform_cross_correlation(padded_image, norm_img_b, img_height, img_width)
