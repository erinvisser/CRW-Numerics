import numpy as np
import matplotlib.pyplot as plt

from .solver import solve_equation
from .random_policy import random_alpha, solve_u_alpha

def sweep_deltas(problem,
                 deltas,
                 method="jacobi",
                 tol=1e-6,
                 max_it=10_000,
                 save_snapshots=False):
    """
    Run solve_equation for multiple delta values.

    Returns:
        results: dict
            delta -> dict with keys:
                'u', 'policy', 'hist', 'snapshots', 'iters_snap'
    """
    results = {}

    for d in deltas:
        u, policy, hist, snaps, iters_snap = solve_equation(
            problem,
            method=method,
            delta=d,
            tol=tol,
            max_it=max_it,
            save_snapshots=save_snapshots,
        )

        results[d] = {
            "u": u,
            "policy": policy,
            "hist": hist,
            "snapshots": snaps,
            "iters_snap": iters_snap,
        }

    return results

def run_random_policies(problem, 
                        u_opt: np.ndarray,
                        *,
                        n: int = 50,
                        seed: int = 0,
                        method: str = "gs",
                        tol: float = 1e-6,
                        max_it: int = 10_000):
    """
    Sample n random policies alpha, solve u_alpha for each, compare to u_opt.

    Returns dict with:
      - errors_inf: (n,) array of ||u_alpha - u_opt||_inf over interior
      - iters:      (n,) array of iterations used
      - alphas:     list of alpha arrays (optional but handy)
      - solutions:  list of u_alpha arrays (optional; can be memory heavy)
    """
    rng = np.random.default_rng(seed)

    mask = problem.mask
    g = problem.g
    f = problem.f
    h = problem.h
    kernels = problem.kernels

    errors = np.zeros(n, dtype=float)
    iters = np.zeros(n, dtype=int)

    alphas = []
    solutions = []

    for t in range(n):
        a_seed = int(rng.integers(0, 1_000_000_000))
        alpha = random_alpha(mask, n_actions=len(kernels), seed=a_seed)

        u_a, it_a, _, _ = solve_u_alpha(
            mask, g, f, kernels, alpha,
            h=h, method=method, tol=tol, max_it=max_it,
            save_snapshots=False
        )

        # error on interior only
        diff = u_a - u_opt
        errors[t] = float(np.max(np.abs(diff[mask])))
        iters[t] = int(it_a)

        alphas.append(alpha)
        solutions.append(u_a)

    return {
        "errors_inf": errors,
        "iters": iters,
        "alphas": alphas,
        "solutions": solutions,
    }