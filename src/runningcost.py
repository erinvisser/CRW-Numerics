import numpy as np

def zero_cost(nx: int, ny: int) -> np.ndarray:
    return np.zeros((nx, ny), dtype=float)

def constant_cost(nx: int, ny: int, c: float) -> np.ndarray:
    return np.full((nx, ny), float(c), dtype=float)

def gaussian_cost(nx: int, ny: int, sigma: float) -> np.ndarray:
    cx = nx / 2
    cy = ny / 2
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.exp(-((I - cx)**2 + (J - cy)**2) / (2.0 * sigma**2))