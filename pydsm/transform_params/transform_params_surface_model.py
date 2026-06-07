import numpy as np
from numba.experimental import jitclass


class TransformStaticParametersSurfaceModel:
    def __init__(self, m_free, m_fxd, p_pillars, k_betas_free, k_betas_pos):
        self.m_free = m_free
        self.m_fxd = m_fxd
        self.m = self.m_free + self.m_fxd

        self.p_pillars = p_pillars
        self.k_betas_free = k_betas_free
        self.k_betas_pos = k_betas_pos
        self.k_betas = self.k_betas_free + self.k_betas_pos

        self._k = int(self.p_pillars + self.m + self.k_betas)
        self._k_unknown = int(2*self.m_free +self.m + self.p_pillars + self.k_betas)
        self._idx_c = np.arange(self.m_free)
        self._idx_T = np.arange(self.m_free, 2 * self.m_free)
        self._idx_Q = np.arange(2 * self.m_free, 2 * self.m_free + self.m)
        self._idx_H = np.arange(2 * self.m_free + self.m, 2 * self.m_free + self.m + self.p_pillars)
        self._idx_betas = np.arange(2 * self.m_free + self.m + self.p_pillars,
                                    2 * self.m_free + self.m + self.p_pillars + self.k_betas)

        self._idx_Q_H = np.arange(2 * self.m_free, 2 * self.m_free + self.m + self.p_pillars)
        self._idx_betas_free = np.arange(2 * self.m_free + self.m + self.p_pillars,
                                         2 * self.m_free + self.m + self.p_pillars + self.k_betas_free)
        self._idx_betas_pos = np.arange(2 * self.m_free + self.m + self.p_pillars + self.k_betas_free,
                                        2 * self.m_free + self.m + self.p_pillars + self.k_betas)

        self._seasonal_betas = np.array([[ 0.15209905],
                                        [ 0.87062539],
                                        [ 0.29587792],
                                        [ 0.02493672],
                                        [-0.07752371],
                                        [-0.08177866],
                                        [ 0.05919807],
                                        [-0.00586872]]) *0.95


    def from_pos_lb_to_r(self, a,  lb):
        return np.log(a - lb)

    def from_r_to_pos_lb(self, a,  lb):
        return np.exp(a) + lb

    def from_pos_to_r(self, a):
        return np.log(a)

    def from_r_to_pos(self, a):
        return np.exp(a)

    def from_min1_plus1_to_r(self, a):
        return -np.log((1-a)/a)  #np.log((1+a)/(1-a))/2
        # lb = 0.5
        # ub = 1
        # tmp = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
        # return lb + (ub - lb) * (tmp / 2 + 0.5)

    def from_r_to_min1_plus1(self, a):
        return np.exp(a)/(1+np.exp(a))  #(np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))
        # lb = 0.5
        # ub = 1
        # tmp = 2 * (a - lb) / (ub - lb) - 1
        # return np.log((1 + tmp) / (1 - tmp)) / 2

    def stack(self, c, T, Q, H_pillars, betas):
        # Stack parameters
        stacked_params = np.concatenate((c[:self.m_free].flatten(), np.diag(T)[:self.m_free], np.diag(Q),
                                         H_pillars.flatten(), betas.flatten()), axis=0).reshape(self._k_unknown,1)
        return stacked_params

    def unstack(self, stacked_params):
        # Split stacked param vector
        c = np.concatenate((stacked_params[self._idx_c].flatten(), np.zeros(self.m_fxd, dtype=np.float64))).reshape(self.m,1)
        diagT = np.concatenate((stacked_params[self._idx_T].flatten(), np.ones(self.m_fxd, dtype=np.float64)))
        diagQ = stacked_params[self._idx_Q].flatten()
        H_pillars = stacked_params[self._idx_H].reshape(self.p_pillars, 1)
        betas = stacked_params[self._idx_betas].reshape(self.k_betas, 1)

        # vectors to matrices
        T = np.diag(diagT)
        Q = np.diag(diagQ)

        return c, T, Q, H_pillars, betas

    def transform(self, stacked_params):
        transformed_stacked_params = np.zeros((self._k_unknown,1), dtype=np.float64)
        # transform stacked param vector
        transformed_stacked_params[self._idx_c] = stacked_params[self._idx_c]
        transformed_stacked_params[self._idx_T] = self.from_min1_plus1_to_r(stacked_params[self._idx_T])  # diagT
        transformed_stacked_params[self._idx_Q_H] = self.from_pos_to_r(stacked_params[self._idx_Q_H])  # diagQ, H_pillars
        transformed_stacked_params[self._idx_betas_free] = stacked_params[self._idx_betas_free]  # free betas, no link needed
        transformed_stacked_params[self._idx_betas_pos] = self.from_pos_to_r(stacked_params[self._idx_betas_pos])  # pos betas
        return transformed_stacked_params

    def untransform(self, transformed_stacked_params):
        stacked_params = np.zeros((self._k_unknown, 1), dtype=np.float64)
        # untransform stacked param vector
        stacked_params[self._idx_c] = transformed_stacked_params[self._idx_c]
        stacked_params[self._idx_T] = self.from_r_to_min1_plus1(transformed_stacked_params[self._idx_T])  # diagT
        stacked_params[self._idx_Q_H] = self.from_r_to_pos(transformed_stacked_params[self._idx_Q_H])  # diagQ, H_pillars
        stacked_params[self._idx_betas_free] = transformed_stacked_params[self._idx_betas_free]  # free betas, no link needed
        stacked_params[self._idx_betas_pos] = self.from_r_to_pos(transformed_stacked_params[self._idx_betas_pos])  # pos betas
        return stacked_params

    def mle_inits(self, p_pillars_ones:int=0):
        # Set inits
        scale = 0.25
        c = np.zeros((self.m_free,1), dtype=np.float64)
        diagT = np.ones((self.m_free,1), dtype=np.float64)*0.97
        diagQ = np.ones((self.m, 1), dtype=np.float64) * scale *scale *scale
        H_pillars = np.ones((self.p_pillars,1), dtype=np.float64) * scale  * scale
        H_pillars[:p_pillars_ones] = 1
        try:
            betas = self._seasonal_betas[:self.k_betas].reshape(self.k_betas,1)
        except:
            betas = np.ones((self.k_betas, 1), dtype=np.float64) * 0.2



        # Transform inits
        trans_diagT = self.from_min1_plus1_to_r(np.copy(diagT))
        trans_diagQ = self.from_pos_to_r(np.copy(diagQ))
        trans_H_pillars = self.from_pos_to_r(np.copy(H_pillars))

        # Stack transformed inits
        transformed_stacked_params = np.concatenate((c, trans_diagT, trans_diagQ, trans_H_pillars, betas), axis=0)
        return transformed_stacked_params

    def mle_inits_random_grid_search(self, M):
        # Set inits
        lb_c = np.ones((self.m_free,1), dtype=np.float64) * -0.01
        ub_c = np.ones((self.m_free, 1), dtype=np.float64) * 0.01

        ub_diagT = self.from_min1_plus1_to_r(np.ones((self.m_free,1), dtype=np.float64)*0.99)
        lb_diagT = self.from_min1_plus1_to_r(np.ones((self.m_free, 1), dtype=np.float64) * 0.97)

        lb_diagQ = self.from_pos_to_r(np.ones((self.m, 1), dtype=np.float64) * 0.001)
        ub_diagQ = self.from_pos_to_r(np.ones((self.m, 1), dtype=np.float64) * 0.1)

        lb_Hpillars = self.from_pos_to_r(np.ones((self.p_pillars, 1), dtype=np.float64) * 0.01)
        ub_Hpillars = self.from_pos_to_r(np.ones((self.p_pillars, 1), dtype=np.float64) * 10)

        lb_betas_free = np.ones((self.k_betas_free, 1), dtype=np.float64) *-1
        ub_betas_free = np.ones((self.k_betas_free, 1), dtype=np.float64) * 1

        lb_betas_pos = self.from_pos_to_r(np.ones((self.k_betas_pos, 1), dtype=np.float64) * 0.2)
        ub_betas_pos = self.from_pos_to_r(np.ones((self.k_betas_pos, 1), dtype=np.float64) * 0.6)

        lb = np.concatenate((lb_c, lb_diagT, lb_diagQ, lb_Hpillars, lb_betas_free, lb_betas_pos), axis=0)
        ub = np.concatenate((ub_c, ub_diagT, ub_diagQ, ub_Hpillars, ub_betas_free, ub_betas_pos), axis=0)

        # np.random.seed(7)
        pars = np.random.uniform(lb, ub, size=(self._k_unknown, M)).T.reshape(M, self._k_unknown,1)
        return pars

# p = 3  #DONOTCHTNGE
# m_free = 5
# m_fxd = 8
#
# rsp = TransformStaticParametersSurfaceModel(m_free, m_fxd, p, 1, 3)
# inits = rsp.mle_inits()
# rsp.mle_inits_random_grid_search(M=100)
# untransformed = rsp.untransform(inits)
# transformed = rsp.transform(untransformed)
# unstacked = rsp.unstack(untransformed)
# stacked = rsp.stack(*unstacked)
#
# print(inits)
# print(untransformed)
# print(transformed)
# print(unstacked)
# print(stacked)