import numpy as np

def nanmax_abs(a: np.ndarray) -> float:
    return float(np.nanmax(np.abs(a)))

def assert_shape(name: str, arr: np.ndarray, shape: tuple[int, int]) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
