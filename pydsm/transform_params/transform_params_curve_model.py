import numpy as np
from numba.experimental import jitclass


class TransformStaticParametersCurveModel:
    def __init__(self, p, m_free, m_fixed, k_lmbdas, k_betas):

        self.p = p
        self.m_act = m_free + m_fixed
        self.m_free = m_free 
        self.m_fixed = m_fixed
        self._k_lmbdas = k_lmbdas
        self.k_betas = k_betas
        self._k = int(m_free + m_free + self.m_act + p + k_lmbdas + k_betas)

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

    def from_r_to_min1_plus1(self, a):
        return np.exp(a)/(1+np.exp(a))  #(np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))


    def stack(self, c, T, Q, H, lmbdas, betas):
        k = int(self._k)
        # Stack parameters
        stacked_params = np.concatenate((c.flatten()[:self.m_free], np.diag(T)[:self.m_free], np.diag(Q),
                                         np.diag(H), lmbdas.flatten(), betas.flatten()), axis=0).reshape(k,1)
        return stacked_params

    def unstack(self, stacked_params):
        p = int(self.p)
        m = int(self.m_free)

        # Split stacked param vector
        c = np.concatenate((stacked_params[:m].flatten(), np.zeros(self.m_fixed, dtype=np.float64))).reshape(self.m_act, 1)
        diagT = np.concatenate((stacked_params[m:(m+m)].flatten(), np.ones(self.m_fixed, dtype=np.float64))).reshape(self.m_act, 1)
        diagQ = stacked_params[(m+m):(m+m+self.m_act)]
        diagH = stacked_params[(m+m+self.m_act):(m+m+self.m_act+p)]
        lmbdas = stacked_params[(m+m+self.m_act+p):(m+m+self.m_act+p+self._k_lmbdas)].reshape(self._k_lmbdas,1)
        betas = stacked_params[(m+m+self.m_act+p+self._k_lmbdas):(m+m+self.m_act+p+self._k_lmbdas+self.k_betas)].reshape(self.k_betas,1)

        # vectors to matrices
        T = np.zeros((self.m_act,self.m_act), dtype=np.float64)
        Q = np.zeros((self.m_act,self.m_act), dtype=np.float64)
        H = np.zeros((p,p), dtype=np.float64)

        np.fill_diagonal(T, diagT)
        np.fill_diagonal(Q, diagQ)
        np.fill_diagonal(H, diagH)

        return c, T, Q, H, lmbdas, betas

    def transform(self, stacked_params):
        p = int(self.p)
        m = int(self.m_free)
        k = int(self._k)
        transformed_stacked_params = np.zeros((k,1), dtype=np.float64)

        # transform stacked param vector
        transformed_stacked_params[:m] = stacked_params[:m]  # c, no link needed
        transformed_stacked_params[m:(m+m)] = self.from_min1_plus1_to_r(stacked_params[m:(m+m)])  # diagT
        transformed_stacked_params[(m+m):(m+m+self.m_act)] = self.from_pos_to_r(stacked_params[(m+m):(m+m+self.m_act)])  # diagQ
        transformed_stacked_params[(m+m+self.m_act):(m+m+self.m_act+p)] = self.from_pos_to_r(stacked_params[(m+m+self.m_act):(m+m+self.m_act+p)])  # diagH
        transformed_stacked_params[(m+m+self.m_act+p):(m+m+self.m_act+p+self._k_lmbdas)] = self.from_pos_to_r(stacked_params[(m+m+self.m_act+p):(m+m+self.m_act+p+self._k_lmbdas)])  # lmbdas
        transformed_stacked_params[(m+m+self.m_act+p+self._k_lmbdas):(m+m+self.m_act+p+self._k_lmbdas+self.k_betas)] = stacked_params[(m+m+self.m_act+p+self._k_lmbdas):(m+m+self.m_act+p+self._k_lmbdas+self.k_betas)] #betas
        return transformed_stacked_params

    def untransform(self, transformed_stacked_params):
        p = int(self.p)
        m = int(self.m_free)
        k = int(self._k)
        stacked_params = np.zeros((k, 1), dtype=np.float64)

        # untransform stacked param vector
        stacked_params[:m] = transformed_stacked_params[:m]  # c, no link needed
        stacked_params[m:(m+m)] = self.from_r_to_min1_plus1(transformed_stacked_params[m:(m+m)])  # diagT
        stacked_params[(m+m):(m+m+self.m_act)] = self.from_r_to_pos(transformed_stacked_params[(m+m):(m+m+self.m_act)])  # diagQ
        stacked_params[(m+m+self.m_act):(m+m+self.m_act+p)] = self.from_r_to_pos(transformed_stacked_params[(m+m+self.m_act):(m+m+self.m_act+p)])  # diagH
        stacked_params[(m+m+self.m_act+p):(m+m+self.m_act+p+self._k_lmbdas)] = self.from_r_to_pos(transformed_stacked_params[(m+m+self.m_act+p):(m+m+self.m_act+p+self._k_lmbdas)])  # lmbdas
        stacked_params[(m+m+self.m_act+p+self._k_lmbdas):(m+m+self.m_act+p+self._k_lmbdas+self.k_betas)] = transformed_stacked_params[(m+m+self.m_act+p+self._k_lmbdas):(m+m+self.m_act+p+self._k_lmbdas+self.k_betas)] # betas
        return stacked_params

    def mle_inits(self):
        p = int(self.p)
        m = int(self.m_free)

        # Set inits
        scale = 0.25
        c = np.zeros((m,1), dtype=np.float64)
        diagT = np.ones((m,1), dtype=np.float64)*0.98
        diagQ = np.ones((self.m_act,1), dtype=np.float64)*scale*scale
        diagH = np.ones((p,1), dtype=np.float64)*scale*scale
        lmbdas = np.ones((self._k_lmbdas,1), dtype=np.float64)
        betas = np.ones((self.k_betas,1), dtype=np.float64) /10

        # Transform inits
        trans_diagT = self.from_min1_plus1_to_r(np.copy(diagT))
        trans_diagQ = self.from_pos_to_r(np.copy(diagQ))
        trans_diagH = self.from_pos_to_r(np.copy(diagH))

        # Stack transformed inits
        transformed_stacked_params = np.concatenate((c, trans_diagT, trans_diagQ, trans_diagH, lmbdas, betas), axis=0)
        return transformed_stacked_params

# p = 1  #DONOTCHTNGE
# m = 2
#
# rsp = TransformStaticParametersCurveModel(p, m, 2, 1, 2)
# inits = rsp.mle_inits()
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