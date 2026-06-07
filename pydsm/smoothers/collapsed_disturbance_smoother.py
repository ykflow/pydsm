import numpy as np
from numba import jit


# %%
### MULTIVARIATE DISTURBANCE SMOOTHER
@jit(nopython=True, cache=True)
def disturbanceSmoother(v, invF, K, Q, H_star, r, N):
    ## CONFIG 
    ########################################
    # Dimensions
    n, p, m = H_star

    # Disturbance Smoother Matrices/Vectors     # At time t:
    epsilon_hat = np.zeros(shape=(n, p, 1))  # Smoothed observation error (p x 1)
    eta_hat = np.zeros(shape=(n, m, 1))  # Smoothed state innovation (m x 1)
    covEpsilon_hat = np.zeros(shape=(n, p, p))
    covEta_hat = np.zeros(shape=(n, m, m))


    ## Disturbance smoother recursions
    ########################################
    for t in range(n - 1, -1, -1):
        # Disturbance updates
        epsilon_hat[t] = H_star @ (invF[t] @ v[t] - K[t].T @ r[t])
        eta_hat[t] = Q  @ r[t]
        covEpsilon_hat[t] = H_star - H_star @ (invF[t] + K[t].T @ N[t] @ K[t]) @ H_star
        covEta_hat[t] = Q - Q  @ N[t] @ Q

    return epsilon_hat, covEpsilon_hat, eta_hat, covEta_hat