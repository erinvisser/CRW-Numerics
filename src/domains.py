import numpy as np

def rectangle_domain(nx: int, ny: int) -> np.ndarray:
    """Rectangle interior points TRUE, boundary points FALSE."""
    mask = np.ones((nx, ny), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask

def circle_domain(nx: int, ny: int, center: tuple[float, float], radius: float) -> np.ndarray:
    """Circle interior points TRUE, everything else FALSE."""
    cx, cy = center
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = (I - cx)**2 + (J - cy)**2 <= radius**2

    # boundary points set FALSE
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
    return inside

def interior_points(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(i), int(j)) for i, j in pts]
