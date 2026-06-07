import numpy as np
from numba import jit
from numba.typed import List
from utilities.special_numba_funcs import random_normal
from scipy.stats import norm


def sim_zero_mean_state(Phi, Q, H, a1, P1, N, k, S):
    a_plus = np.zeros((N, k, S))
    eta_plus = norm.rvs(0, 1, size=(N, k, S))
    eps_plus = np.linalg.cholesky(H) @ norm.rvs(0, 1, size=(N, k, S))

    eta_plus[1:] = np.sqrt(Q) @ eta_plus[1:]
    a_plus[0] = a1.copy() + np.sqrt(Q) @ P1 @ eta_plus[0]
    for t in range(N-1):
        a_plus[t+1] = Phi @ a_plus[t] + eta_plus[t]

    y_plus = a_plus + eps_plus
    return y_plus, a_plus


@jit(nopython=True, cache=True)
def sim_nonlinear_measurement(sigma2, Zf, f, FIXED, betas, N, p, k, Isigma2):
    y = np.zeros((N, p, 1))
    for t in range(N):
        y[t] = Zf(f[t], FIXED[t], betas, p, k) + Isigma2[t]* np.sqrt(sigma2) * random_normal(p).reshape(p,1)
    return y


# def simulation_smoother(y_tilde, a_plus,  Zf, FIXEDx, Phi, c, Q, H, a1, P):
#     ## CONFIG
#     ########################################
#     # Dimensions
#     N, k, _ = a_plus.shape
#
#     a_hat_star = kalmanSmoother(y_tilde, Z, T, R, H, Q, a1, P1)[0]
#     a_hat_sim = a_plus_s + a_hat_star.reshape(n ,m)
#
#     return a_hat_sim

# def simple_simulation_smoother(y, Z, T, R, H, Q, a1, P1):
#     ## CONFIG
#     ########################################
#     # Dimensions
#     n, p, m = y.shape[0], y.shape[1], R.shape[0]
#     eta_sim = np.zeros((n ,m) ) *np.nan
#
#     eps_hat = disturbanceSmoother(y, Z, T, R, H, Q, a1, P1)[0]
#     y_plus, a_plus, eps_plus = uncStateSSM(n, m, p, 1, Z, T, R, H, Q, P1, a1=y[0])
#     eps_hat_plus = disturbanceSmoother(y_plus, Z, T, R, H, Q, a1, P1)[0]
#
#     eps_sim = eps_plus - eps_hat_plus + eps_hat
#     a_hat_sim = y - eps_sim
#
#     for t in range(n -1):
#         eta_sim[t] = a_hat_sim[ t +1] - a_hat_sim[t]
#
#     return a_plus, a_hat_sim, eps_sim, eta_sim
#
#
