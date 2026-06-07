import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, pinv
from numba import jit
from utilities.special_numba_funcs import logdet


# @jit(nopython=True, cache=True)
def collapsed_extended_kalman_filter_tvp_q(yx, Zf, FIXEDx, betas, c, Phi, Q, H, a1, P1, N, k, Wx, Ix, Sq, Iq, Jac=None):

    def dZdf(Zf, a, FIXED, betas, p, k):
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

    ## CONFIG
    ########################################
    # Dimensions
    y_star = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
    H_star = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
    h = np.zeros(shape=(N+1, 1), dtype=np.float64)
    a = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
    P = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
    a_t = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
    P_t = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
    LL = np.zeros(shape=(N, 1), dtype=np.float64)

    # Intitalization and connstant/default objects
    J = dZdf if Jac is None else Jac
    a[0], P[0] = a1, P1
    invH = np.diag(1/np.diag(H))
    log2pi = np.log(2*np.pi)
    noise = np.eye(k) * 1e-6  ## THIS IS THE ONE 1e-6
    eye_m = np.eye(k)
    h[0] = 0.001
    lmbda1 = 0.025
    lmbda2 = 0.97
    omega = np.diag(H).mean() * (1- lmbda1 - lmbda2) /-2
    ## FILTER RECURSIONS
    ########################################
    for t in range(N):
        # print(t)
        # Collapsing
        px = yx[t].shape[0]
        # Hx = np.diag(np.diag(H) + h[t]*0)
        Hx = Wx[t] @ (H + h[t]*0) @ Wx[t].T
        invHx = np.diag(1/np.diag(Hx))
        Qx = Q #* (eye_m - Sq * Iq[t])
        Qx = np.diag(np.diag(Qx) + h[t])

        Zfx = Zf(a[t], FIXEDx[t], betas, px, k)
        Jx = J(Zf, a[t], FIXEDx[t], betas, px, k)
        eps_star = yx[t] - Zfx
        tmp0 = Jx.T @ np.diag(1/(eps_star.flatten() **2))
        H_star[t] = inv(tmp0 @ Jx + noise)
        y_star[t] = a[t] + H_star[t] @ tmp0 @ eps_star

        # Prediction error decomposition
        vx = (y_star[t] - a[t]) * Ix[t]
        epsx = eps_star - Jx @ vx
        Fx = P[t] + H_star[t]
        invFx = inv(Fx)
        tmp1 = P[t] @ invFx

        # Filter step
        a_t[t] = a[t] + tmp1 @ vx
        P_t[t] = P[t] - tmp1 @ P[t]

        # Compute log-likelihood
        logdet_Hx = np.log(np.diag(Hx)).sum()
        ll_f = -0.5 * (k * log2pi + logdet(Fx) + vx.T @ invFx @ vx)
        ll_y = -0.5 * ((px - k) * log2pi + logdet_Hx + epsx.T @ invHx @ epsx
                       )  #(invHx/var_epsx)
        LL[t] = (ll_f + ll_y + 0.5 * logdet(H_star[t])
                 ) * Ix[t]
        # print(LL[t])
        # if np.abs(LL[t]) >= 1e4 or np.isnan(LL[t]):
        #     LL[t]

        # Update step
        a[t+1] = c + Phi @ a_t[t]
        P[t+1] = Phi @ P_t[t] @ Phi.T + Qx
        P[t+1] = (P[t+1] + P[t+1].T) / 2  # Enforce symmetry of predicted covariance matrix
        h[t+1] = omega + lmbda2 * h[t] + lmbda1*(epsx**2).mean()
    plt.plot(h[:-1]), plt.show()
    return a[:N], P[:N], a_t[:N], P_t[:N], LL[:N]

# @jit(nopython=True, cache=True)
# def Zf(a, FIXED, betas, p, m):
#     f0, f1, f2, f3 = a.flatten()
#     t = FIXED
#     # np.exp(f1*t) * (f2*np.sin(2*np.pi*t) + f2*np.cos(2*np.pi*t))
#     # ones = np.ones((p,1))
#     # return np.exp(ones * a.sum()).reshape(p,1)
#     tau = np.linspace(0, 2, p).reshape(p,1)
#     T = t + tau
#     S = f2*np.sin(2*np.pi*T) + f3*np.cos(2*np.pi*T)
#     return f0 + np.exp(-np.exp(f1) *tau) *S #np.exp(FIXED @ a) #np.exp(FIXED @ a)
#
# @jit(nopython=True, cache=True)
# def dZdf(Zf, a, FIXED, betas, p, m):
#     h = 1e-8 * (np.fabs(a) + 1e-8)  # Find stepsize
#     h = np.maximum(h, 5e-6)
#
#     Jac = np.zeros((p,m), dtype=np.float64)
#     for i in range(m):
#         h_i_min = a.copy()
#         h_i_plus = a.copy()
#         h_i_min[i] -= h[i]
#         h_i_plus[i] += h[i]
#         Jac[:, i] = ((Zf(h_i_plus, FIXED, betas, p, m) - Zf(h_i_min, FIXED, betas, p, m))/(2*h[i])).flatten()
#     # print(Jac == FIXED.T)
#     return Jac
#     # return FIXED.T
#
#
# def sim_ssm(Z, T, c, R, sigma2H, sigma2Q, n, burn_in=500):
#     p, m = Z.shape
#     size = n + burn_in
#     y = np.zeros((size, p, 1))
#     alpha = np.zeros((size, m, 1))
#     np.random.seed(0)
#     eps = np.random.normal(size=(size, p, 1)) * np.sqrt(sigma2H)
#     np.random.seed(4)
#     eta = np.random.normal(size=(size, m, 1)) * np.sqrt(sigma2Q)
#
#     alpha[0] = eta[0]
#     for t in range(size-1):
#         alpha[t+1] = c + T @ alpha[t] + R @ eta[t+1]
#         y[t+1] =Zf(alpha[t+1], t, np.array([]), p, m) + eps[t+1]
#
#     return alpha[burn_in:], y[burn_in:]
#
#
#
# from numba import typeof
# from utilities.special_numba_funcs import logdet, locate_missings
# from numba.typed import List
# import pandas as pd
# import matplotlib.pyplot as plt
# n, p = 100, 24
# m =4
#
# c = np.zeros((m,1))
# sigma2H = 0.001
# sigma2Q = 0.001
# T = np.eye(m)*1
# R = np.eye(m)
# Sq = np.eye(m)
# Iq = np.zeros(n)
# Iq[50:75] = 1
#
# np.random.seed(7)
# Z = np.abs(np.random.normal(size=(p,m)))
# Z /= Z.sum(axis=1).reshape(p,1)
#
# state, y = sim_ssm(Z, T, c, R, sigma2H, sigma2Q, n, burn_in=500)
#
# pd.DataFrame(y.reshape(n,p)).plot(figsize=(20,10), legend=False)
# plt.show()
#
# pd.DataFrame(state.reshape(n,m)).plot(figsize=(20,10), legend=False)
# plt.show()
#
# H = np.eye(p) * sigma2H
# Q = np.eye(m) * sigma2Q
# ones_q = np.ones(m)
# ones_q[1:] = 0
# Sq = np.diag(ones_q)
#
# a1, P1 = state[0], Q
#
# yx = List()
# Ix = np.zeros(n)
# Wx = List()
# zeros_p = np.zeros((p, 1), dtype=np.float64)
# FIXEDx = List()
# for t in range(n):
#     loc_missing, nobs_missing, loc_complete, nobs_complete = locate_missings(np.copy(y[t]))
#     mask = loc_complete.flatten()
#
#     W = np.eye(p, dtype=np.float64)[mask]
#     y_tmp = np.copy(y[t])
#     y_tmp[loc_missing.flatten()] = 0  # put NA to zero to enable matrix algebra
#
#     # Create star (x) matrices
#     if nobs_complete > 0:
#         Ix[t] = 1.
#         yx.append(W @ y_tmp)
#         Wx.append(W)
#         # FIXEDx.append(W @ Z)
#         FIXEDx.append(t)
#     else:
#         Ix = 0.
#         yx.append(zeros_p)
#         Wx.append(np.eye(p, dtype=np.float64))
#         # FIXEDx.append(Z)
#         FIXEDx.append(t)
#
# results = collapsed_extended_kalman_filter_tvp_q(yx, Zf, FIXEDx, np.array([]),
#                                            c, T,
#                                            Q, H, a1, P1, n, m, Wx, Ix, Sq, Iq)
# print(results[-1].sum())
#
# fig, ax = plt.subplots(figsize=(20,10))
# pd.DataFrame(state.reshape(n,m)).plot(ax=ax, color='gray', legend=False)
# pd.DataFrame(results[2].reshape(n,m)).plot(ax=ax, color='red', legend=False)
# plt.show()
#
# print(np.square(state - results[0]).mean())
#
# from time import time
# start = time()
# M = 50
# for i in range(M):
#     print(i)
#     results = collapsed_extended_kalman_filter_tvp_q(yx, Zf, FIXEDx, np.array([]),
#                                                c, T,
#                                                Q, H, a1, P1, n, m, Wx, Ix, Sq, Iq)
# end = time()
#
# print('avg time (s)', 1/((end - start)/M))
#
