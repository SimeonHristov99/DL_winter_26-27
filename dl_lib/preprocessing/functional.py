import numpy as np


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    min_val = np.min(array)
    max_val = np.max(array)

    if max_val != min_val:
        return (array - min_val) / (max_val - min_val)
    
    return np.zeros_like(array)


def standardize(array: np.ndarray) -> np.ndarray:
    mean = np.mean(array)
    std = np.std(array)

    return (array - mean) / std if std != 0 else np.zeros_like(array)