from numba import jit
from numba.typed import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filters.extended_kalman_filter import extended_kalman_filter as EKF
from filters.collapsed_extended_kalman_filter import collapsed_extended_kalman_filter as CEKF
from utilities.special_numba_funcs import locate_missings
from time import time


@jit(nopython=True, cache=True)
def Zf(a, FIXED, betas, p, m):
    # ones = np.ones((p,1))
    # return np.exp(ones * a.sum()).reshape(p,1)
    return np.exp(FIXED @ a)

@jit(nopython=True, cache=True)
def jac(Zf, f, FIXED, betas, p, m):
    # x, tau, T, Z = FIXED
    return FIXED * np.exp(FIXED @ f)
def sim_ssm(Z, T, c, R, sigma2H, sigma2Q, n, burn_in=500):
    p, m = Z.shape
    size = n + burn_in
    y = np.zeros((size, p, 1))
    alpha = np.zeros((size, m, 1))
    # np.random.seed(0)
    eps = np.random.normal(size=(size, p, 1)) * np.sqrt(sigma2H)
    # np.random.seed(4)
    eta = np.random.normal(size=(size, m, 1)) * np.sqrt(sigma2Q)

    alpha[0] = eta[0]
    for t in range(size-1):
        alpha[t+1] = c + T @ alpha[t] + R @ eta[t+1]
        y[t+1] =Zf(alpha[t+1], Z, np.array([]), p, m) + eps[t+1]

    return alpha[burn_in:], y[burn_in:]




n = 100
M = 2
# list_m = [1, 5, 10, 25, 50]
# list_p = [1, 10, 100, 250, 500]
# #
list_m = [5]
list_p = [10]

gains_table = pd.DataFrame(columns=list_p, index=list_m)

for m in list_m:
    for p in list_p:
        if p >= m:
            c = np.zeros((m,1))
            sigma2H = 0.01
            sigma2Q = 0.01
            T = np.eye(m)*0.975
            R = np.eye(m)

            # np.random.seed(7)
            Z = np.abs(np.random.normal(size=(p,m)))
            Z /= Z.sum(axis=1).reshape(p,1)

            state, y = sim_ssm(Z, T, c, R, sigma2H, sigma2Q, n, burn_in=500)

            pd.DataFrame(y.reshape(n,p)).plot(figsize=(20,10), legend=False)
            plt.show()

            H = np.eye(p) * sigma2H
            Q = np.eye(m) * sigma2Q

            a1, P1 = state[0]*0.98, np.eye(m, dtype=np.float64)/5

            yx = List()
            Ix = np.zeros(n)
            Wx = List()
            zeros_p = np.zeros((p, 1), dtype=np.float64)
            FIXEDx = List()
            for t in range(n):
                loc_missing, nobs_missing, loc_complete, nobs_complete = locate_missings(np.copy(y[t]))
                mask = loc_complete.flatten()

                W = np.eye(p, dtype=np.float64)[mask]
                y_tmp = np.copy(y[t])
                y_tmp[loc_missing.flatten()] = 0  # put NA to zero to enable matrix algebra

                # Create star (x) matrices
                if nobs_complete > 0:
                    Ix[t] = 1.
                    yx.append(W @ y_tmp)
                    Wx.append(W)
                    FIXEDx.append(W @ Z)
                else:
                    Ix = 0.
                    yx.append(zeros_p)
                    Wx.append(np.eye(p, dtype=np.float64))
                    FIXEDx.append(Z)

            results_ekf = EKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)
            results_cekf = CEKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)
            results_ekf = EKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)
            results_cekf = CEKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)
            results_ekf = EKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)
            results_cekf = CEKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)

            fig, ax = plt.subplots(figsize=(20,10))
            ax.plot(results_cekf[-3] -results_ekf[-1])
            # ax.plot(results_ekf[-1])
            plt.tight_layout()
            plt.show()


            # fig, ax = plt.subplots(figsize=(20,10))
            # ax.plot(results_cekf[-1])
            # ax.plot(results_ekf[-1])
            # plt.tight_layout()
            # plt.show()
            #
            # fig, ax = plt.subplots(figsize=(20,10))
            # ax.plot(results_ekf[0].reshape(n,m), color='red')
            # ax.plot(results_cekf[0].reshape(n,m), color='black', linestyle='dotted')
            # plt.tight_layout()
            # plt.show()
            #
            # fig, ax = plt.subplots(figsize=(20, 10))
            # ax.plot(state.reshape(n, m), color='red')
            # ax.plot(results_cekf[0].reshape(n, m), color='black', linestyle='dotted')
            # plt.tight_layout()
            # plt.show()

            start = time()
            for i in range(M):
                print(i)
                results = EKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)
            end = time()
            time_ekf = 1/((end - start)/M)

            start = time()
            for i in range(M):
                print(i)
                results = CEKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix, jac)
            end = time()
            time_cekf = 1 / ((end - start) / M)
            gain = time_cekf / time_ekf

        else:
            gain = np.nan

        gains_table.loc[m,p] = gain
        print(m,p)


# fig, ax = plt.subplots(figsize=(20,10))
# pd.DataFrame(state.reshape(n,m)).plot(ax=ax, color='gray', legend=False)
# pd.DataFrame(results_ekf[2].reshape(n,m)).plot(ax=ax, color='red', legend=False)
# plt.show()
#
# fig, ax = plt.subplots(figsize=(20,10))
# pd.DataFrame(state.reshape(n,m)).plot(ax=ax, color='gray', legend=False)
# pd.DataFrame(results_cekf[2].reshape(n,m)).plot(ax=ax, color='red', legend=False)
# plt.show()
#
#
#
#
# ###################################
# n,p,m = 100, 500, 1
# c = np.zeros((m,1))
# sigma2H = 0.01
# sigma2Q = 0.01
# T = np.eye(m)*0.975
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
# results_ekf = EKF(yx, Zf, FIXEDx, np.array([]), Q, H, a1, P1, n, m, Wx, Ix)
# results_cekf = CEKF(yx, Zf, FIXEDx, np.array([]), c, T, Q, H, a1, P1, n, m, Wx, Ix)
