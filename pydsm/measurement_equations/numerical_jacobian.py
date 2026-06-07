import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def JacTwoSided(Zf, a, FIXED, betas, p, k):
    h = 1e-8 * (np.fabs(a) + 1e-8)  # Find stepsize
    h = np.maximum(h, 5e-6)

    Jac = np.zeros((p, k), dtype=np.float64)
    for i in range(k):
        h_i_min = a.copy()
        h_i_plus = a.copy()
        h_i_min[i] -= h[i]
        h_i_plus[i] += h[i]
        Jac[:, i] = ((Zf(h_i_plus, FIXED, betas, p, k) - Zf(h_i_min, FIXED, betas, p, k)) / (2 * h[i])).flatten()
    return Jac