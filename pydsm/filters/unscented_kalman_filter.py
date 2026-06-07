import numpy as np
from numpy.linalg import inv, cholesky
from numba import jit
from utilities.special_numba_funcs import logdet

# @jit(nopython=True)
# def cholesky(A):
#     """Perform Cholesky decomposition on symmetric, positive definite matrix."""
#     n = A.shape[0]
#     L = np.zeros_like(A, dtype=np.float64)
#
#     # Perform the Cholesky decomposition
#     for row in range(n):
#         for col in range(row+1):
#             tmp_sum = 0.0
#             for j in range(col):
#                 tmp_sum += L[row, j] * L[col, j]
#             if (row == col):
#                 # diag elts.
#                 L[row, col] = (A[row, row] - tmp_sum)**.5
#             else:
#                 L[row, col] = (1.0 / L[col, col] * (A[row, col] - tmp_sum))
#     return L

@jit(nopython=True, cache=True)
def unscented_kalman_filter(yx, Zf, FIXEDx, betas, c, Phi, Q, H, a1, P1, N, k, Wx, Ix, Sq=None, Iq=None, jac=None):
    ## CONFIG
    ########################################
    # Dimensions
    a = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
    P = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
    a_t = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
    P_t = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
    LL = np.zeros(shape=(N, 1), dtype=np.float64)

    # Intitalization and constant/default objects
    a[0], P[0] = a1, P1
    # invH = np.diag(1/np.diag(H))
    l = 3 - k
    L = 2 * k + 1
    u = np.sqrt(k + l)
    omega = np.ones(L) / (2*(k+l))
    omega[0] = l / (k + l)
    log2pi = np.log(2 * np.pi)
    ## FILTER RECURSIONS
    ########################################
    for t in range(N):
        # Prediction error decomposition
        px = yx[t].shape[0]
        Hx = Wx[t] @ H @ Wx[t].T
        # invHx = Sx[t] @ invH @ Sx[t].T
        sqrtP = cholesky(P[t])
        X = np.concatenate((a[t], a[t] + u*sqrtP, a[t] - u*sqrtP), axis=1).T.reshape(L, k, 1)

        mux = np.zeros((px,1))
        ZfXx = np.zeros((L, px,1))
        for i in range(L):
            ZfXx[i] = Zf(X[i], FIXEDx[t], betas, px, k)
            mux += omega[i] * ZfXx[i]

        vXx = ZfXx - mux
        vx = (yx[t] - mux) * Ix[t]
        Pavx = np.zeros((k, px))
        Pvxvx = np.zeros((px, px))
        for i in range(L):
            Pavx += omega[i] * (X[i] - a[t]) @ vXx[i].T
            Pvxvx += omega[i] * vXx[i] @ vXx[i].T

        Pvxvx += Hx
        invPvxvx = inv(Pvxvx)
        tmp1 = Pavx @ invPvxvx

        # Filter step
        a_t[t] = a[t] + tmp1 @ vx
        P_t[t] = P[t] - tmp1 @ Pavx.T

        # Compute log-likelihood
        LL[t] = -0.5 * (px * log2pi + logdet(Pvxvx) + vx.T @ invPvxvx @ vx) * Ix[t]

        # Update step
        a[t+1] = c + Phi @ a_t[t]
        P[t+1] = Phi @ P_t[t] @ Phi.T + Q
        P[t+1] = (P[t+1] + P[t+1].T) / 2  # Enforce symmetry of predicted covariance matrix

    return a[:N], P[:N], a_t[:N], P_t[:N], LL[:N]

#
# @jit(nopython=True, cache=True)
# def Zf(a, FIXED, betas, p, m):
#     # ones = np.ones((p,1))
#     # return np.exp(ones * a.sum()).reshape(p,1)
#     return FIXED @ a
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
# n, p = 2500, 20
# m = 20
#
# c = np.zeros((m,1))
# sigma2H = 10
# sigma2Q = 5
# T = np.eye(m)
# R = np.eye(m)
# Z = np.eye(p)
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
#
# # y[500:1500, (0, 2, 4)] = np.nan
#
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
#     W = np.eye(p)[mask]
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
#         Wx.append(np.eye(p))
#         FIXEDx.append(Z)
#
# results = unscented_kalman_filter(yx, Zf, FIXEDx, np.array([]), Q, H, a1, P1, n, m, Wx, Ix)
# print(results[-1].sum())
#
# fig, ax = plt.subplots(figsize=(20,10))
# pd.DataFrame(state.reshape(n,m)).plot(ax=ax, color='gray', legend=False)
# pd.DataFrame(results[2].reshape(n,m)).plot(ax=ax, color='red', legend=False)
# plt.show()
#
#
# from time import time
# start = time()
# M = 50
# for i in range(M):
#     print(i)
#     results = unscented_kalman_filter(yx, Zf, FIXEDx, np.array([]), Q, H, a1, P1, n, m, Wx, Ix)
# end = time()
#
# print('avg time (s)', 1/((end - start)/M))