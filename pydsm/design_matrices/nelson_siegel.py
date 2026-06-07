import numpy as np

class NelsonSiegelDesignMatrix:
    def __init__(self, lmbda: np.array,  tau: np.array):
        self.lmbda = lmbda
        self.tau = tau

    @staticmethod
    def _n_lmbdas():
        return 1

    @staticmethod
    def _m_states():
        return 2

    def accept_lmbdas(self):
        return True

    def make_design_matrix(self):
        n, p = self.tau.shape
        slope = (1 - np.exp(-self.lmbda * self.tau)) / (self.lmbda * self.tau)
        curvature = slope - np.exp(-self.lmbda * self.tau)
        Z = np.concatenate((slope.reshape(n,p,1), curvature.reshape(n,p,1)), axis=2)
        return Z

# from random import choices
# p = 10
# n = 2
# lmbdas = np.array([1])
# tau = np.array(choices(np.linspace(1/13, 31, 10), k=n*p)).reshape(n,p)
#
# from time import time
# mat = np.linspace(1/13, 31, 10)
# start = time()
# cns = NelsonSiegelDesignMatrix(lmbdas, tau)
# Z = cns.make_design_matrix()
# end = time()
# print(end - start)
# print(Z)
#

