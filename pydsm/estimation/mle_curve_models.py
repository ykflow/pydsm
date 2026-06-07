import numpy as np
from scipy.optimize import minimize, approx_fprime
from design_matrices.curve_design_matrix_builder import CurveDesignMatrixBuilder
from filters.depreciated.kalman_filter_X import kalman_filter_X
from utilities.special_numba_funcs import locate_missings
from transform_params.transform_params_curve_model import TransformStaticParametersCurveModel
from numpy.linalg import inv
from numba.typed import List
from time import time


class MLECurveModels:
    def __init__(self, y: np.array, tau, timeline, builder: CurveDesignMatrixBuilder,
                 tol_mle=1e-7, max_iter_mle=500, mle_inits=None, diffuse: bool=False,):

        n, p, _ = y.shape
        self._n = n
        self._p = p
        self.y = y
        self.tau = tau
        self.timeline = timeline

        self.cdmb = builder
        self.tol_mle = tol_mle
        self.max_iter_mle = max_iter_mle
        self.mle_inits = mle_inits
        self.diffuse = diffuse

        self._m = self.cdmb._m
        self._m_free = self.cdmb._m_free
        self._m_fixed = self.cdmb._m_fixed
        self._k_betas = self.cdmb._k_betas
        self._k_lmbdas = self.cdmb._k_lmbdas
        self._tsp = TransformStaticParametersCurveModel(1, self._m_free, self._m_fixed, self._k_lmbdas, self._k_betas)
        self._R = np.eye(self._m, dtype=np.float64)

        self._initialize = self.diffuse_inits # if self.diffuse else self.stationary_inits
        self._burn_in = 5

        self._mle_optim_results = None
        self._mle_params = None

    @staticmethod
    def prepare_fixed_input_kf(y):
        n, p, _ = y.shape
        yx = List()
        Ix = np.zeros(n)
        Wx = List()
        zeros_p = np.zeros((p, 1), dtype=np.float64)
        for t in range(n):
            loc_missing, nobs_missing, loc_complete, nobs_complete = locate_missings(np.copy(y[t]))
            mask = loc_complete.flatten()

            W = np.eye(p)[mask]
            y_tmp = np.copy(y[t])
            y_tmp[loc_missing.flatten()] = 0  # put NA to zero to enable matrix algebra

            # Create star (x) matrices
            if nobs_complete > 0:
                Ix[t] = 1.
                yx.append(W @ y_tmp)
                Wx.append(W)
            else:
                Ix[t] = 0.
                yx.append(zeros_p)
                Wx.append(np.eye(p))
        return yx, Ix, Wx

    @staticmethod
    def stationary_inits(c, T, Q):
        m = T.shape[0]
        eye_m = np.eye(m, dtype=np.float64)
        eye_mm = np.eye(m * m, dtype=np.float64)
        a1 = inv(eye_m - T) @ c
        vecQ = np.copy(Q.reshape(m * m, 1))
        vecP1 = inv(eye_mm - np.kron(T, T)) @ vecQ
        P1 = vecP1.reshape(m, m)
        return a1, P1  # *100

    @staticmethod
    def diffuse_inits(c, T, Q):
        m = T.shape[0]
        eye_m = np.eye(m, dtype=np.float64)
        a1 = np.zeros((m, 1))
        P1 = eye_m * 1e6
        return a1, P1

    def config_mle(self):
        # self._filter = kalman_filter_tvp_h if self._with_variance_curve else kalman_filter
        self._filter = kalman_filter_X
        self.yx, self.Ix, self.Wx = self.prepare_fixed_input_kf(np.copy(self.y))

        if self.mle_inits is None:
            self._mle_inits = self._tsp.mle_inits().flatten()
        else:
            self._mle_inits = self.mle_inits

    def _build_kwargs_kf(self, transformed_params):
        k = transformed_params.shape[0]
        transformed_params = transformed_params.reshape(k, 1)
        untransformed_params = self._tsp.untransform(transformed_params)
        c, T, Q, H_variances, lmbdas, betas = self._tsp.unstack(untransformed_params)
        Z, Xbeta = self.cdmb.build_design_matrix(None, None, lmbdas, betas)
        H = np.eye(self._p) * H_variances
        kwargs_ll = (Z, Xbeta, c, T, Q, H)
        return kwargs_ll, Z, Xbeta, c, T, Q, H, lmbdas, betas, untransformed_params

    def loglike(self, kwargs_ll, sum_loglike=True):
        try:
            Z, Xbeta, c, T, Q, H = kwargs_ll
            a1, P1 = self._initialize(c, T, Q)
            kwargs_kf = (self.yx, Z, Xbeta, c, T, self._R, Q, H, a1, P1, self._n, self._m, self.Wx, self.Ix)
            a, P, a_t, P_t, LL = self._filter(*kwargs_kf)
            if sum_loglike == True:
                return LL[self._burn_in:].sum()
            else:
                return LL[self._burn_in:]
        except:
            if sum_loglike == True:
                return np.ones((1)) * - np.inf
            else:
                return np.ones((self._n, 1))[self._burn_in:] * - np.inf

    @staticmethod
    def __print_params(untransformed_params: np.array):
        k = untransformed_params.shape[0]
        try:
            print(untransformed_params.reshape(int(k / 3), 3).round(4))
        except:
            try:
                print(untransformed_params.reshape(int(k / 2), 2).round(4))
            except:
                try:
                    print(untransformed_params.reshape(int(k / 1), 1).round(4))
                except:
                    pass

    def objf(self, transformed_params):
        kwargs_ll, Z, Xbeta, c, T, Q, H, lmbdas, betas, untransformed_params = self._build_kwargs_kf(transformed_params)
        sum_ll = self.loglike(kwargs_ll=kwargs_ll)
        self.__print_params(untransformed_params)
        return sum_ll

    def estimate(self):
        self.config_mle()
        nobs = np.invert(np.isnan(self.y)).sum()
        func = lambda params: - self.objf(params) / nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        start = time()
        self._mle_optim_results = minimize(fun=func, x0=self._mle_inits, method='SLSQP',
                                           jac=grad,
                                           # tol=self.tol_mle,
                                           options={'maxiter': self.max_iter_mle})
        end = time()
        self._run_time = end - start

    def get_untransformed_params(self):
        transformed_params = self._mle_optim_results.x
        k = transformed_params.shape[0]
        transformed_params = transformed_params.reshape(k, 1)
        untransformed_params = self._tsp.untransform(transformed_params)
        return untransformed_params

    def from_untransformed_params_to_transformed_params(self, untransformed_params):
        transformed_params = self._tsp.transform(untransformed_params)
        return transformed_params

    def gather_results(self):
        print(self._mle_optim_results)
        transformed_estimates = self._mle_optim_results.x
        kwargs_ll, Z, Xbeta, c, T, Q, H, lmbdas, betas, untransformed_params = self._build_kwargs_kf(transformed_estimates)
        a1, P1 = self._initialize(c, T, Q)
        return untransformed_params, lmbdas, betas, self.yx, Z, Xbeta, c, T, self._R, Q, H, a1, P1, self.Wx, self.Ix

