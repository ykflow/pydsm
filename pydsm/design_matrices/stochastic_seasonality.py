import numpy as np


class StochasticSeasonalityDesignMatrix:
    def __init__(self, k: int, t: np.array):
        self.k = k
        self.t = t

    # @staticmethod
    def _n_lmbdas(self):
        return self.k*2

    def _m_states(self):
        return self.k*2

    def accept_lmbdas(self):
        return True

    def make_design_matrix(self):
        n, p = self.t.shape
        Z = np.zeros((n, p, self.k*2))
        for i in range(self.k):
            f = (2*np.pi * self.t * (i+1)).reshape(n,p,1)
            loc = np.arange(2 * i, 2 * (i + 1))
            s = np.concatenate((np.sin(f), np.cos(f)), axis=2)
            Z[:, :, loc] = s
        return Z


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


