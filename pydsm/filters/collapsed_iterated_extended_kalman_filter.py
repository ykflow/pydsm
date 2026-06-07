# import numpy as np
# from numpy.linalg import pinv as inv
# from numba import jit
# from utilities.special_numba_funcs import logdet
#
# # @jit(nopython=True, cache=True)
# def collapsed_iterated_extended_kalman_filter(yx, Zf, FIXEDx, betas, c, Phi, Q, H, a1, P1, N, k, Wx, Ix, Jac=None):
#
#     def dZdf(Zf, a, FIXED, betas, p, k):
#         h = 1e-8 * (np.fabs(a) + 1e-8)  # Find stepsize
#         h = np.maximum(h, 5e-6)
#
#         Jac = np.zeros((p, k), dtype=np.float64)
#         for i in range(k):
#             h_i_min = a.copy()
#             h_i_plus = a.copy()
#             h_i_min[i] -= h[i]
#             h_i_plus[i] += h[i]
#             Jac[:, i] = ((Zf(h_i_plus, FIXED, betas, p, k) - Zf(h_i_min, FIXED, betas, p, k)) / (2 * h[i])).flatten()
#         return Jac
#
#     def iterate(a_start, y, invH, Zf, J, FIXED, betas, p, k):
#         f_plus = a_start.copy()
#         eps2 = 1
#         i = 0
#         noise = 1e-6
#         cond = False
#         tol = 0.00001
#         max_iter = 1
#         while cond == False:
#             print(i, eps2)
#             f_i = f_plus.copy()
#             Zf_i = Zf(f_i, FIXED, betas, p, k)
#             Jx_i = J(Zf, f_i, FIXED, betas, p, k)
#             tmp0 = Jx_i.T @ invH
#             H_star_i = inv(tmp0 @ Jx_i + noise)
#             delta = + H_star_i @ tmp0 @ (y - Zf_i)
#             f_plus = f_i + delta
#             eps2 = (delta.T @ delta).flatten()[0] ** .5
#             i = i + 1
#             cond = eps2 <= tol or i >= max_iter
#
#         return f_plus, H_star_i, Zf_i, Jx_i
#
#
#     ## CONFIG
#     ########################################
#     # Dimensions
#     y_star = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
#     H_star = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
#     a = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
#     P = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
#     a_t = np.zeros(shape=(N + 1, k, 1), dtype=np.float64)
#     P_t = np.zeros(shape=(N + 1, k, k), dtype=np.float64)
#     LL = np.zeros(shape=(N, 1), dtype=np.float64)
#
#     # Intitalization and connstant/default objects
#     J = dZdf if Jac is None else Jac
#     a[0], P[0] = a1, P1
#     a_t[0], P_t[0] = a1, P1
#     invH = np.diag(1/np.diag(H))
#     log2pi = np.log(2*np.pi)
#     noise = np.eye(k) * 1e-6  ## THIS IS THE ONE 1e-6
#     ## FILTER RECURSIONS
#     ########################################
#     for t in range(N):
#         # print(t)
#         # Collapsing
#         px = yx[t].shape[0]
#         Hx = Wx[t] @ H @ Wx[t].T
#         invHx = Wx[t] @ invH @ Wx[t].T
#         y_star[t], H_star[t], Zfx, Jx = iterate(a[t], yx[t], invHx, Zf, J, FIXEDx[t], betas, px, k)
#
#         # Zfx = Zf(a[t], FIXEDx[t], betas, px, k)
#         # Jx = J(Zf, a[t], FIXEDx[t], betas, px, k)
#         # tmp0 = Jx.T @ invHx
#         # H_star[t] = inv(tmp0 @ Jx + noise)
#         # y_star[t] = a[t] + H_star[t] @ tmp0 @ (yx[t] - Zfx)
#
#         # Prediction error decomposition
#         vx = (y_star[t] - a[t]) * Ix[t]
#         epsx = yx[t] - Zfx - Jx @ vx
#         Fx = P[t] + H_star[t]
#         invFx = inv(Fx)
#         tmp1 = P[t] @ invFx
#
#         # Filter step
#         a_t[t] = a[t] + tmp1 @ vx
#         P_t[t] = P[t] - tmp1 @ P[t]
#
#         # Compute log-likelihood
#         logdet_Hx = np.log(np.diag(Hx)).sum()
#         ll_f = -0.5 * (k * log2pi + logdet(Fx) + vx.T @ invFx @ vx)
#         ll_y = -0.5 * ((px - k) * log2pi + logdet_Hx + epsx.T @ invHx @ epsx
#                        )  # (invHx/var_epsx)
#         LL[t] = (ll_f + ll_y + 0.5 * logdet(H_star[t])
#                  ) * Ix[t]
#
#         # Update step
#         a[t+1] = c + Phi @ a_t[t]
#         P[t+1] = Phi @ P_t[t] @ Phi.T + Q
#         P[t+1] = (P[t+1] + P[t+1].T) / 2  # Enforce symmetry of predicted covariance matrix
#
#     return a[:N], P[:N], a_t[:N], P_t[:N], LL[:N]
#
# @jit(nopython=True, cache=True)
# def Zf(a, FIXED, betas, p, m):
#     # f0, f1, f2, f3 = a.flatten()
#     # t = FIXED
#     # np.exp(f1*t) * (f2*np.sin(2*np.pi*t) + f2*np.cos(2*np.pi*t))
#     # ones = np.ones((p,1))
#     # return np.exp(ones * a.sum()).reshape(p,1)
#     # tau = np.linspace(0, 2, p).reshape(p,1)
#     # T = t + tau
#     # S = f2*np.sin(2*np.pi*T) + f3*np.cos(2*np.pi*T)
#     return np.tan(np.exp(-np.exp(FIXED @ a))) #np.exp(FIXED @ a)
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
# m = 4
#
# c = np.zeros((m,1))
# sigma2H = 0.01
# sigma2Q = 0.01
# T = np.eye(m)*0.985
# R = np.eye(m)
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
#
# a1, P1 = state[0]*0.99, np.eye(m, dtype=np.float64)
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
#         # FIXEDx.append(t)
#     else:
#         Ix = 0.
#         yx.append(zeros_p)
#         Wx.append(np.eye(p, dtype=np.float64))
#         FIXEDx.append(Z)
#         # FIXEDx.append(t)
#
# results = collapsed_iterated_extended_kalman_filter(yx, Zf, FIXEDx, np.array([]),
#                                            c, T,
#                                            Q, H, a1, P1, n, m, Wx, Ix)
# print(results[-1].sum())
#
# fig, ax = plt.subplots(figsize=(20,10))
# pd.DataFrame(state.reshape(n,m)).plot(ax=ax, color='gray', legend=False)
# pd.DataFrame(results[2].reshape(n,m)).plot(ax=ax, color='red', legend=False)
# plt.show()
#
# print(np.square(state - results[0]).mean(axis=0))
#
# # # from time import time
# # # start = time()
# # # M = 50
# # # for i in range(M):
# # #     print(i)
# # #     results = collapsed_iterated_extended_kalman_filter(yx, Zf, FIXEDx, np.array([]),
# # #                                                c, T,
# # #                                                Q, H, a1, P1, n, m, Wx, Ix)
# # # end = time()
# # #
# # # print('avg time (s)', 1/((end - start)/M))
#
#
#
# # @jit(nopython=True, cache=True)
# # def Zf(a, FIXED, betas, p, m):
# #     f1, f2 = a.flatten()
# #     x = FIXED.copy()
# #     return (f1 * x)/ (f2 + x)
# #
# # @jit(nopython=True, cache=True)
# # def dZdf(Zf, a, FIXED, betas, p, m):
# #     h = 1e-8 * (np.fabs(a) + 1e-8)  # Find stepsize
# #     h = np.maximum(h, 5e-6)
# #
# #     Jac = np.zeros((p,m), dtype=np.float64)
# #     for i in range(m):
# #         h_i_min = a.copy()
# #         h_i_plus = a.copy()
# #         h_i_min[i] -= h[i]
# #         h_i_plus[i] += h[i]
# #         Jac[:, i] = ((Zf(h_i_plus, FIXED, betas, p, m) - Zf(h_i_min, FIXED, betas, p, m))/(2*h[i])).flatten()
# #     # print(Jac == FIXED.T)
# #     return Jac
# #     # return FIXED.T
# #
# # def gauss_newton(f0, y, invH, Zf, J, FIXED, betas, p, k, max_iter=20, tol=0.0001):
# #     f_plus = f0.copy()
# #     eps2 = 1
# #     i = 0
# #     noise = 1e-6
# #     cond = False
# #     while cond == False:
# #         print(i, eps2)
# #         f_i = f_plus.copy()
# #         Zf_i = Zf(f_i, FIXED, betas, p, k)
# #         Jx_i = J(Zf, f_i, FIXED, betas, p, k)
# #         tmp0 = Jx_i.T @ invH
# #         H_star_i = inv(tmp0 @ Jx_i + noise)
# #         f_plus = f_i + H_star_i @ tmp0 @ (y - Zf_i)
# #         eps = (f_plus - f_i)
# #         eps2 = (eps.T @ eps).flatten()[0] **.5
# #         i = i + 1
# #
# #         cond = eps2 <= tol or i >= max_iter
# #
# #     return f_plus
# #
# #
# # f0 = np.array([[0.9, 0.2]]).reshape(2,1)
# # x = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740]).reshape(7,1)
# # y = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317]).reshape(7,1)
# # invH = np.eye(7) *10
# # betas = None
# # p, k = 7, 2
# #
# #
# # dZdf(Zf, f0, x, betas, p, k)
# #
# # print(gauss_newton(f0, y, invH, Zf, dZdf, x, betas, p, k, max_iter=20, tol=0.001))
# #
#
