import numpy as np

from .boundary import apply_dirichlet
from .domains import interior_points

def random_alpha(mask: np.ndarray, n_actions: int = 2, seed: int | None = None) -> np.ndarray:
    """
    alpha[i,j] on interior (mask True), -1 on boundary/outside.
    """
    rng = np.random.default_rng(seed)
    alpha = np.full(mask.shape, -1, dtype=int)
    pts = np.argwhere(mask)
    alpha[pts[:, 0], pts[:, 1]] = rng.integers(0, n_actions, size=len(pts))
    return alpha

def avg_alpha(u_src: np.ndarray, i: int, j: int, kernels: list[dict], alpha: np.ndarray) -> float:
    """
    Average using the kernel selected by alpha[i,j].
    """
    K = kernels[int(alpha[i, j])]
    return (
        K["up"]    * u_src[i, j + 1] +
        K["down"]  * u_src[i, j - 1] +
        K["left"]  * u_src[i - 1, j] +
        K["right"] * u_src[i + 1, j]
    )

def solve_u_alpha(mask: np.ndarray,
                  g: np.ndarray,
                  f: np.ndarray,
                  kernels: list[dict],
                  alpha: np.ndarray,
                  *,
                  h: float = 1.0,
                  method: str = "gs",      # "gs" or "jacobi"
                  tol: float = 1e-6,
                  max_it: int = 10_000,
                  save_snapshots: bool = False,
                  snapshot_freq: int = 20):
    """
    Solve fixed-policy equation on interior:
        u(i,j) = Ave_{alpha(i,j)}(u)(i,j) - h^2 f(i,j)
    with u=g on mask False points.

    Convergence: ||u^{n+1} - u^n||_inf < tol.

    Returns:
        u, it, snapshots, iters_snap
    """
    assert mask.shape == g.shape == f.shape == alpha.shape
    nx, ny = mask.shape

    # init slightly above boundary max
    u = np.full((nx, ny), float(np.max(g[~mask]) + 1e-3))
    apply_dirichlet(u, g, mask)

    pts = interior_points(mask)

    snapshots = []
    iters_snap = []

    for it in range(max_it):
        u_prev = u.copy()
        u_src = u_prev if method == "jacobi" else u

        for i, j in pts:
            avg = avg_alpha(u_src, i, j, kernels, alpha)
            u[i, j] = avg - (h ** 2) * f[i, j]

        apply_dirichlet(u, g, mask)

        if save_snapshots and (it % snapshot_freq == 0):
            snapshots.append(u.copy())
            iters_snap.append(it)

        if np.max(np.abs(u - u_prev)) < tol:
            break

    return u, it, snapshots, iters_snap