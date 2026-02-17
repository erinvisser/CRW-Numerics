import numpy as np

from .operators import best_avg, residual_inf
from .boundary import apply_dirichlet
from .domains import interior_points

def solve_equation(problem,
                  method: str = "jacobi",    # "jacobi" or "gs" 
                  delta: float = 1.0,  
                  tol: float = 1e-6,
                  max_it: int = 10_000,
                  save_snapshots: bool = False,
                  snapshot_freq: int = 20):
    """
    Solve: M(u) = h^2 f with BC u=g on boundary.
    Returns (u, policy, res_hist, snapshots, iters_snap)
    """
    nx, ny = problem.nx, problem.ny
    mask, g, f, h, kernels = problem.mask, problem.g, problem.f, problem.h, problem.kernels

    # set up contraction equation
    if not (0.0 < delta <= 1.0):
        raise ValueError("contraction (delta) must be in (0, 1].")

    # initialize u as slightly above boundary max, so it is guaranteed to be wrong
    u = np.full((nx, ny), float(np.max(g[~mask]) + 1e-3))
    apply_dirichlet(u, g, mask)

    pts = interior_points(mask)
    policy = np.full((nx, ny), -1, dtype=int) # walker choice

    res_hist = [] # initialize residuals
    snapshots = [] # initialize for movie
    iters_snap = []

    for it in range(max_it):
        u_old = u.copy()

        for i, j in pts: # loop through all interior points
            src = u_old if method == "jacobi" else u # decide whether to use old (jacobi) or new (gauss-seidel)
            best_average, best_k = best_avg(src, i, j, kernels) # pick best average
            u[i, j] = delta * best_average - (h**2) * f[i, j]
            policy[i, j] = best_k

        apply_dirichlet(u, g, mask)

        # always compute the true residual from og eqn
        res_true = residual_inf(u, h, f, mask, kernels)

        entry = {"it": it, "res_true": res_true}

        # only store delta residual when doing contraction for jacobi
        if method == "jacobi" and not np.isclose(delta, 1.0, atol=1e-12):
            res_delta = residual_inf(u, h, f, mask, kernels)
            entry["res_delta"] = res_delta

        res_hist.append(entry)

        if save_snapshots and (it % snapshot_freq == 0):
            snapshots.append(u.copy())
            iters_snap.append(it)

        # check convergence
        if np.max(np.abs(u-u_old)) < 1e-6:
            print(f"Converged in {it} iterations.")
            break

    return u, policy, res_hist, snapshots, iters_snap