import numpy as np
from typing import Callable, Sequence, Tuple, Optional

BCFn = Callable[[int, int, int, int], float]

def build_boundary(nx: int, ny: int, mask: np.ndarray, bc: BCFn) -> np.ndarray:
    """
    Return array for g.
    bc(i, j, nx, ny) -> float
    """
    g = np.zeros((nx, ny), dtype=float)
    bd = ~mask
    idx = np.argwhere(bd)
    for i, j in idx:
        g[i, j] = bc(int(i), int(j), nx, ny)
    return g

def apply_dirichlet(u: np.ndarray, g: np.ndarray, mask: np.ndarray) -> None:
    u[~mask] = g[~mask]

def constant_boundary(c: float):
    def bc(i, j, nx, ny):
        return float(c)
    return bc

def affine_boundary(i, j, nx, ny):
    if i == 0:
        return 1.0
    if i == nx - 1:
        return 0.0
    if j == 0 or j == ny - 1:
        return 1.0 - i/(nx - 1)
    return 0.0

def sinusoidal_boundary(
    amp: float = 1.0,
    kx: int = 2,
    ky: int = 2,
    offset: float = 0.0,
) -> BCFn:
    def bc(i: int, j: int, nx: int, ny: int) -> float:
        x = i / (nx - 1) if nx > 1 else 0.0
        y = j / (ny - 1) if ny > 1 else 0.0

        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            return float(offset + amp * (np.sin(2*np.pi*kx*x) + np.cos(2*np.pi*ky*y)) / 2.0)
        return 0.0
    return bc

def checkerboard_boundary(
    v0: float = -1.0,
    v1: float = 1.0,
    period: int = 4,
) -> BCFn:
    period = max(1, int(period))

    def bc(i: int, j: int, nx: int, ny: int) -> float:
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            key = (i // period + j // period) % 2
            return float(v0 if key == 0 else v1)
        return 0.0

    return bc

def door_boundary(
    side: str = "right",   # "left" | "right" | "top" | "bottom"
    door_value: float = 1.0,
    wall_value: float = 0.0,
    door_center_frac: float = 0.5,
    door_width_frac: float = 0.2,
) -> BCFn:
    side = side.lower()

    def in_door(idx: int, n: int) -> bool:
        center = door_center_frac * (n - 1)
        halfw  = 0.5 * door_width_frac * (n - 1)
        return abs(idx - center) <= halfw

    def bc(i: int, j: int, nx: int, ny: int) -> float:
        if side == "left" and i == 0:
            return door_value if in_door(j, ny) else wall_value
        if side == "right" and i == nx - 1:
            return door_value if in_door(j, ny) else wall_value
        if side == "bottom" and j == 0:
            return door_value if in_door(i, nx) else wall_value
        if side == "top" and j == ny - 1:
            return door_value if in_door(i, nx) else wall_value

        # other edges are walls
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            return wall_value

        return 0.0

    return bc

def random_chunk_boundary(
    nx: int,
    ny: int,
    seed: Optional[int] = None,
    nseg_left: int = 3,
    nseg_right: int = 3,
    nseg_bottom: int = 3,
    nseg_top: int = 3,
    values: Sequence[float] = (-1.0, 1.0),
) -> BCFn:
    rng = np.random.default_rng(seed)

    def make_segments(n: int, nseg: int) -> np.ndarray:
        # assign each boundary index to a segment id
        nseg = max(1, min(n, nseg))
        cuts = np.linspace(0, n, nseg + 1, dtype=int)
        seg_id = np.zeros(n, dtype=int)
        for s in range(nseg):
            seg_id[cuts[s]:cuts[s+1]] = s
        # random value per segment
        seg_vals = rng.choice(np.array(values, dtype=float), size=nseg)
        return seg_vals[seg_id]

    left_vals   = make_segments(ny, nseg_left)    # indexed by j
    right_vals  = make_segments(ny, nseg_right)
    bottom_vals = make_segments(nx, nseg_bottom)  # indexed by i
    top_vals    = make_segments(nx, nseg_top)

    def bc(i: int, j: int, nx: int, ny: int) -> float:
        if i == 0:
            return float(left_vals[j])
        if i == nx - 1:
            return float(right_vals[j])
        if j == 0:
            return float(bottom_vals[i])
        if j == ny - 1:
            return float(top_vals[i])
        return 0.0  # not used for interior; safe default

    return bc
