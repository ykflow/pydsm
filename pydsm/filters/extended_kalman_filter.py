import numpy as np
from numpy.linalg import inv
from numba import jit
from utilities.special_numba_funcs import logdet

@jit(nopython=True, cache=True)
def extended_kalman_filter(yx, Zf, FIXEDx, betas, c, Phi, Q, H, a1, P1, N, k, Sx, Ix, Jac=None):

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
    ## FILTER RECURSIONS
    ########################################
    for t in range(N):
        # Prediction error decomposition
        px = yx[t].shape[0]
        Hx = Sx[t] @ H @ Sx[t].T
        invHx = Sx[t] @ invH @ Sx[t].T
        invP = inv(P[t])
        Zfx = Zf(a[t], FIXEDx[t], betas, px, k)
        Jx = J(Zf, a[t], FIXEDx[t], betas, px, k)

        vx = (yx[t] - Zfx) * Ix[t]
        Fx = Jx @ P[t] @ Jx.T + Hx
        invFx = invHx - invHx @ Jx @ inv(invP + Jx.T @ invHx @ Jx) @ Jx.T @ invHx #inv(Fx)
        tmp1 = P[t] @ Jx.T @ invFx

        # Filter step
        a_t[t] = a[t] + tmp1 @ vx
        P_t[t] = P[t] - tmp1 @ Jx @ P[t]

        # Compute log-likelihood
        LL[t] = -0.5 * (px * log2pi + logdet(Fx) + vx.T @ invFx @ vx) * Ix[t]

        # Update step
        a[t+1] = c + Phi @ a_t[t]
        P[t+1] = Phi @ P_t[t] @ Phi.T + Q
        P[t+1] = (P[t+1] + P[t+1].T) / 2  # Enforce symmetry of predicted covariance matrix

    return a[:N], P[:N], a_t[:N], P_t[:N], LL[:N]

# @jit(nopython=True, cache=True)
# def Zf(a, FIXED, betas, p, m):
#     # ones = np.ones((p,1))
#     # return np.exp(ones * a.sum()).reshape(p,1)
#     return np.exp(FIXED @ a) #np.exp(FIXED @ a)
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
#         y[t+1] =Zf(alpha[t+1], Z, np.array([]), p, m) + eps[t+1]
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
# n, p = 100, 100
# m = 10
#
# c = np.zeros((m,1))
# sigma2H = 0.01
# sigma2Q = 0.01
# T = np.eye(m)*0.975
# R = np.eye(m)
#
# np.random.seed(7)
# Z = np.abs(np.random.normal(size=(p,m)))
# Z /= Z.sum(axis=1).reshape(p,1)
# Xbeta = np.zeros((n,p,1))
#
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
#
# a1, P1 = np.zeros((m,1), dtype=np.float64), np.eye(m, dtype=np.float64)*10**7
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
#         FIXEDx.append(W @ Z)
#     else:
#         Ix = 0.
#         yx.append(zeros_p)
#         Wx.append(np.eye(p, dtype=np.float64))
#         FIXEDx.append(Z)
#
# results = extended_kalman_filter(yx, Zf, FIXEDx, np.array([]), Q, H, a1, P1, n, m, Wx, Ix)
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
#     results = extended_kalman_filter(yx, Zf, FIXEDx, np.array([]), Q, H, a1, P1, n, m, Wx, Ix)
# end = time()
#
# print('avg time (s)', 1/((end - start)/M))