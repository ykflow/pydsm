import numpy as np
from numba import jit
from numpy.linalg import inv


@jit(nopython=True, cache=True)
def collapsed_kalman_filter_smoother(y, H, Phi, c, Q, a1, P1, Ix):
    ## CONFIG
    ########################################
    # Dimensions
    n, k, _ = y.shape

    # Kalman Smoother Matrices/Vectors  # At time t:
    a = np.zeros(shape=(n + 1, k, 1), dtype=np.float64)
    P = np.zeros(shape=(n + 1, k, k), dtype=np.float64)
    a_t = np.zeros(shape=(n + 1, k, 1), dtype=np.float64)
    P_t = np.zeros(shape=(n + 1, k, k), dtype=np.float64)
    v = np.zeros(shape=(n + 1, k, 1), dtype=np.float64)
    K = np.zeros(shape=(n + 1, k, k), dtype=np.float64)
    invF = np.zeros(shape=(n + 1, k, k), dtype=np.float64)

    a_hat = np.zeros(shape=(n, k, 1), dtype=np.float64)  # Smoothed State Mean (m x 1)
    V = np.zeros(shape=(n, k, k), dtype=np.float64)  # Smoothed State Variance (m x m)
    L = np.zeros(shape=(n, k, k), dtype=np.float64)  # Scaled Kalman gain (m x m)
    r = np.zeros(shape=(n, k, 1), dtype=np.float64)  # Scaled Smoothed Estimator (m x 1)
    N = np.zeros(shape=(n, k, k), dtype=np.float64)  # Scaled Smoothed Estimator Variance (m x m)
    I = np.eye(k)

    ## Kalman filter recursions
    ########################################
    a[0] = a1
    P[0] = P1
    for t in range(n):
        v[t] = (y[t] - a[t]) * Ix[t]
        Ft = P[t] + H[t]
        invF[t] = inv(Ft)
        K[t] = P[t] @ invF[t]

        # Filter step
        a_t[t] = a[t] + K[t] @ v[t]
        P_t[t] = P[t] - K[t] @ P[t]

        # Update step
        a[t + 1] = c + Phi @ a_t[t]
        P[t + 1] = Phi @ P_t[t] @ Phi.T + Q
        P[t + 1] = (P[t + 1] + P[t + 1].T) / 2  # Enforce symmetry of predicted covariance matrix

    ## Kalman smoother recursions
    ########################################
    for t in range(n - 1, -1, -1):
        L[t] = I - K[t]
        r[t - 1] = invF[t] @ v[t] + L[t].T @ r[t]
        N[t - 1] = invF[t] + L[t].T @ N[t] @ L[t]

        # Smoothing Update
        a_hat[t] = a[t] + P[t] @ r[t - 1]
        V[t] = P[t] - P[t] @ N[t - 1] @ P[t]

    return a_hat, V, r, N, L

# %%
# import pandas as pd
# import matplotlib.pyplot as plt
# #%%
# n, p = 100, 1

# state =  np.cumsum(np.random.normal(size=(n,p)), axis=0)
# y = state + np.random.normal(size=(n,p))
# y = pd.DataFrame(y)
# y.plot(figsize=(20,10))
# #%%
# m = p
# Z = T = R = np.eye(p)
# H = np.eye(p)
# Q = np.eye(m)
# a1, P1 = np.zeros((p,1)), np.eye(p)*10**7
# y = y.values.reshape(n, p, 1)
# y[40:60] = np.nan
# #%%
# KF = kalmanFilter(y, Z, T, R, H, Q, a1, P1)
# KFS = kalmanSmoother(y, Z, T, R, H, Q, a1, P1)
# #%%
# fig, ax = plt.subplots(figsize=(20,10))
# #pd.DataFrame(state).plot(ax=ax)
# #pd.DataFrame(KF[0].reshape(n, p)).plot(ax=ax)
# pd.DataFrame(KFS[3].reshape(n, p)).plot(ax=ax)