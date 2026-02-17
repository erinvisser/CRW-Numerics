import numpy as np
from .utils import nanmax_abs

def best_avg(u_src: np.ndarray, i: int, j: int, kernels: list) -> tuple[float, int]:
    """
    Returns (best_avg, best_k) where best_avg = max_k Ave_k(u_src,i,j).
    """
    best = -np.inf
    best_k = -1
    for k, K in enumerate(kernels):
        avg = (
            K["up"]    * u_src[i, j+1] +
            K["down"]  * u_src[i, j-1] +
            K["left"]  * u_src[i-1, j] +
            K["right"] * u_src[i+1, j]
        )
        if avg > best:
            best = avg
            best_k = k
    return float(best), int(best_k)

def compute_M(u: np.ndarray, mask: np.ndarray, kernels: list) -> np.ndarray:
    """
    M(u,x) = max_k (Ave_k(u,x) - u(x)).
    Only defined on interior; else NaN.
    """
    nx, ny = u.shape
    M = np.full((nx, ny), np.nan, dtype=float)
    pts = np.argwhere(mask)
    for i, j in pts:
        i, j = int(i), int(j)
        best_saverage, _ = best_avg(u, i, j, kernels)
        M[i, j] = best_average - u[i, j]
    return M

def residual_inf(u: np.ndarray, h: float, f: np.ndarray, mask: np.ndarray, kernels: list) -> float:
    """
    || M(u) - h^2 f ||_inf
    """
    M = compute_M(u, mask, kernels)

    r = M - (h**2) * f

    r[~mask] = np.nan
    return nanmax_abs(r)