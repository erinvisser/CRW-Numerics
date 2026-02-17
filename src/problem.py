from dataclasses import dataclass
import numpy as np

from .kernels import default_two_walker_kernels
from .utils import assert_shape

@dataclass
class CRWProblem:
    nx: int
    ny: int
    h: float
    mask: np.ndarray
    g: np.ndarray
    f: np.ndarray
    kernels: list

def make_problem(nx: int, ny: int, *, mask: np.ndarray, g: np.ndarray, f: np.ndarray, h: float = 1.0, kernels=None) -> CRWProblem:
    if kernels is None:
        kernels = default_two_walker_kernels()

    assert_shape("mask", mask, (nx, ny))
    assert_shape("g", g, (nx, ny))
    assert_shape("f", f, (nx, ny))

    return CRWProblem(nx=nx, ny=ny, h=h, mask=mask, g=g, f=f, kernels=kernels)
