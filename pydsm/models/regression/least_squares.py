import numpy as np
import pandas as pd
from numpy.linalg import pinv as inv



def truncate(x, decimals=3):
    return np.floor(x * 10 ** decimals) / 10 ** decimals

def add_brackets(x):
    return f'({x})'

def add_signficance_sign(t):
    if (np.abs(t) >= 1.645 and np.abs(t) < 1.96):
        return '$^{*}$'
    elif (np.abs(t) >= 1.96 and np.abs(t) < 2.576):
        return '$^{**}$'
    elif np.abs(t) >= 2.576:
        return '$^{***}$'
    else:
        return ''


class OLS:
    def __init__(self, y: np.array, X: np.array, lag_h:int=None):
        self.y = y
        self.X = X
        self._n, self._p = X.shape
        self.lag_h = np.floor(0.75 * self._n ** (1 / 3)).astype(int) if lag_h is None else lag_h

    def compute_newey_west_standard_errors(self):
        W = np.zeros((self.lag_h + 1, self._p, self._p))
        XSX = np.zeros((self._p, self._p))

        W[0] = self.X.T @ (np.square(self._eps) * self.X)
        for i in range(1, self.lag_h + 1):
            Ri = self._eps[:-i] * self._eps[i:]
            Ui = self.X[i:, ]
            Vi = self.X[:-i, ]
            W[i] = Vi.T @ (Ri * Ui) + Ui.T @ (Ri * Vi)

        for i in range(self.lag_h + 1):
            XSX += (self._n / (self._n - self._p)) * (1 - i / (self.lag_h + 1)) * W[i]

        VCV = inv(self.X.T @ self.X) @ XSX @ inv(self.X.T @ self.X)
        return VCV


    def fit(self):
        self._beta = inv(self.X.T @ self.X) @ self.X.T @ self.y
        self._eps = self.y - self.X @ self._beta
        self._cov_beta = self.compute_newey_west_standard_errors()
        self._se = np.sqrt(np.diag(self._cov_beta)).reshape(self._p, 1)

        RSS = (self._eps.T @ self._eps).flatten()[0]
        TSS = np.square(self.y - self.y.mean()).sum().flatten()[0]
        self._r2 = 1 - RSS/TSS
        self._adj_r2 = 1 - (1 - self._r2) * (self._n - 1) / (self._n - self._p)
        self._RSS = RSS


    def summary(self, idx:list, beta0:float=0):
        df_betas = pd.DataFrame(self._beta, index=idx)
        df_se = pd.DataFrame(self._se, index=idx)
        df_t_stat = (df_betas-beta0) /df_se

        df_betas = df_betas.apply(lambda x: truncate(x)).astype(str)
        df_betas += df_t_stat.map(add_signficance_sign)
        df_se = r'\tiny{(' + df_se.apply(lambda x: truncate(x)).astype(str) + ')}'
        tmp = pd.concat([df_betas, df_se], axis=0, ignore_index=False).loc[idx]
        tmp.loc["AdjR2"] = truncate(self._adj_r2 *100)
        self._summary = tmp.copy()


