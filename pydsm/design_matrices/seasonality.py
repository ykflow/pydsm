import numpy as np


class SeasonalityDesignMatrix:
    def __init__(self, k: int, betas, T: np.array):
        self.k = k
        self.betas = betas
        self.T = T

    # @staticmethod
    def _n_lmbdas(self):
        return self.k*2

    def _m_states(self):
        return self.k*2

    def accept_lmbdas(self):
        return True

    def make_design_matrix(self):
        n, p = self.T.shape
        Xbeta = np.zeros((n, p, 1))
        for i in range(self.k):
            f = (2*np.pi * self.T * (i+1)).reshape(n,p,1)
            loc = np.arange(2 * i, 2 * (i + 1))
            beta1_i, beta2_i = self.betas[loc]
            Xbeta += beta1_i*np.sin(f) + beta2_i*np.cos(f)
        return Xbeta

# from random import choices
# p = 100
# n = 1000
# k = 3
# alphas = np.array([0.2, 0.3, 1, 0.5, -1, -0.5])
# tau = np.array(choices(np.linspace(1/13, 1, 10), k=n*p)).reshape(n,p)
#
# from time import time
# mat = np.linspace(1/13, 31, 10)
# start = time()
# cns = SeasonalityDesignMatrix(k, alphas, tau)
# Z = cns.make_design_matrix()
# end = time()
# print(end - start)
# print(Z)


