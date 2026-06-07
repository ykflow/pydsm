import numpy as np
from scipy.optimize import minimize, approx_fprime
# from filters.extended_kalman_filter import extended_kalman_filter
from filters.collapsed_extended_kalman_filter import collapsed_extended_kalman_filter as CEKF
from filters.collapsed_extended_kalman_filter_tvp_q import collapsed_extended_kalman_filter_tvp_q as CEKFQ
from filters.extended_kalman_filter import extended_kalman_filter as EKF
from filters.unscented_kalman_filter import unscented_kalman_filter as UKF
from transform_params.transform_params_surface_model import TransformStaticParametersSurfaceModel
from measurement_equations.measurement_design_builder import SurfaceMeasurementDesignBuilder
from numba.typed import List
from time import time
import pandas as pd
import matplotlib.pyplot as plt


class MLESurfaceModels:
    def __init__(self, yx: List, FIXEDx: List, Sx: List, Ix: List, builder: SurfaceMeasurementDesignBuilder,
                 tol_mle=1e-7, max_iter_mle=500, collapse:bool=True, unscented:bool=False, mle_inits=None,
                 diffuse: bool=False, find_filter_inits:bool=True,
                 a1:np.array=None, P1:np.array=None, Sq=None, Iq=None):

        self.nobs = np.concatenate(yx, axis=0).shape[0]
        self.yx = yx
        self._N = len(self.yx)
        self.FIXEDx = FIXEDx
        self.Sx = Sx
        self.Ix = Ix
        self.builder = builder
        self.tol_mle = tol_mle
        self.collapse = collapse
        self.unscented = unscented
        self.max_iter_mle = max_iter_mle
        self.mle_inits = mle_inits
        self.diffuse = diffuse

        self._k = self.builder._k_factors
        self._m_free = self.builder._k_free
        self._m_fixed = self.builder._k_fixed
        self._p_pillars = self.builder._p_pillars
        self._k_betas_free = self.builder._k_betas_free
        self._k_betas_pos = self.builder._k_betas_pos
        self._k_betas = self._k_betas_free + self._k_betas_pos
        self._tsp = TransformStaticParametersSurfaceModel(self._m_free, self._m_fixed, self._p_pillars,self._k_betas_free, self._k_betas_pos)
        self._burn_in = 10
        self._mle_optim_results = None
        self._mle_params = None
        self._find_filter_inits = find_filter_inits


        ## DEFAULT INITS
        self.a1 = a1 if a1 is not None else np.zeros((self._k,1))
        self.P1 = P1 if P1 is not None else np.eye(self._k)

        self.Sq = Sq
        self.Iq = Iq

    def find_state_inits(self, fit_daily:bool=False, refit:bool=False):
        # if fit_daily:
        N = self._N
        f_hat = np.zeros((N, self._k))
        beta_hat = np.zeros((N, self._k_betas))
        rmspe = np.zeros((N,))
        np.random.seed(7)
        inits = np.random.normal(size=self._k + self._k_betas, scale=0.1)
        np.random.seed(None)
        for t in range(N):
            print(t)
            px = self.yx[t].shape[0]
            def _objf(params, y, FIXED, p, m, return_rmpse=False):
                try:
                    f = params[:m].reshape(m,1)
                    betas = params[m:]
                    mean = self.builder.Zf(f, FIXED, betas, p, m)
                    error = y - mean
                    loss = (error**2).mean()
                    # loss = np.sqrt(np.nanmean((y ** .5 / mean ** .5 - 1) ** 2))
                    return loss if not return_rmpse else np.sqrt(np.nanmean((y ** .5 / mean ** .5 - 1) ** 2)) *100
                except:
                    loss = np.inf
                    return loss


            func = lambda params: _objf(params, self.yx[t], self.FIXEDx[t], px, self._k)
            grad = lambda params: approx_fprime(params, func, 6.5e-6)

            optim_results = minimize(fun=func, x0=inits, method='SLSQP',
                                               jac=grad, options={'maxiter': 1000})

            if refit:
                optim_results = minimize(fun=func, x0=optim_results.x, method='L-BFGS-B',
                                         jac=grad, options={'maxiter': 1000})

            f_hat[t] = optim_results.x.copy()[:self._k]
            beta_hat[t] = optim_results.x.copy()[self._k:]
            rmspe[t] = _objf(optim_results.x.copy(), self.yx[t], self.FIXEDx[t], px, self._k, True)

        if fit_daily:
            return f_hat, beta_hat, rmspe
        else:
            df_f_hat = pd.DataFrame(f_hat)
            Q1 = df_f_hat.quantile(0.25)
            Q3 = df_f_hat.quantile(0.75)
            IQR = Q3 - Q1
            df_f_hat = df_f_hat[~((df_f_hat < (Q1 - 1.5 * IQR)) | (df_f_hat > (Q3 + 1.5 * IQR))).any(axis=1)].reset_index(drop=True)

            a1 = df_f_hat.iloc[:25].mean().values.reshape(self._k, 1)
            sigma2_eta = df_f_hat.diff().var().values.flatten()
            # sigma2_eta[sigma2_eta >0.1] = 0.1
            P1 = np.diag(sigma2_eta)  # *100

            return a1, P1

    def find_mle_inits(self):
        M = 250
        ll = np.zeros(M)
        inits = self._tsp.mle_inits_random_grid_search(M=M)
        for i in range(M):
            ll[i] = self.objf(inits[i])

        loc = np.argwhere(ll==np.nanmax(ll)).flatten()[0]
        print('FINSIHED FINDING MLE INITS')
        print(np.nanmax(ll))
        return np.random.normal(inits[loc], scale=0.01)

    def config_mle(self):
        self._filter = UKF if self.unscented else CEKF
        self._filter = CEKF if self.collapse else EKF
        self._filter = CEKFQ if self.Sq is not None else CEKF

        if self.mle_inits is None:
            FLAG_WAS_NONE = True
            self.mle_inits = self._tsp.mle_inits(self.builder._p_x).flatten()
            # self._mle_inits = self.find_mle_inits().flatten()
        else:
            self.mle_inits = self.mle_inits.flatten()
            FLAG_WAS_NONE = False

        if self._find_filter_inits:
            self.a1, self.P1 = self.find_state_inits()
            if FLAG_WAS_NONE:
                self.mle_inits[2*self._m_free: 2*self._m_free + self._k] = np.log(np.diag(self.P1).flatten())

    def _build_kwargs_kf(self, transformed_params):
        k = transformed_params.shape[0]
        transformed_params = transformed_params.reshape(k, 1)
        untransformed_params = self._tsp.untransform(transformed_params)
        c, Phi, Q, H_variances, betas = self._tsp.unstack(untransformed_params)
        H = self.builder._build_covH(H_variances)
        kwargs_ll = (c, Phi, Q, H, betas)
        return kwargs_ll, c, Phi, Q, H, betas, untransformed_params

    def loglike(self, kwargs_ll, sum_loglike=True):
        try:
            c, Phi, Q, H, betas = kwargs_ll
            kwargs_kf = (self.yx, self.builder._Zf, self.FIXEDx, betas, c, Phi, Q, H,
                         self.a1, self.P1, self._N, self._k, self.Sx, self.Ix, self.Sq, self.Iq, self.builder._jacobian)
            # a, P, a_t, P_t, LL2 = EKF(*kwargs_kf)
            # a, P, a_t, P_t, LL1 = CEKF(*kwargs_kf)
            try:
                a, P, a_t, P_t, LL = self._filter(*kwargs_kf)[:5]
            except:
                a, P, a_t, P_t, LL = self._filter(*kwargs_kf)[:5]
            # pd.DataFrame(a_t[:, :, 0]).plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
            # pd.DataFrame(a_t[:, 4:7, 0]).plot(figsize=(20, 10), title=f'{np.log(Q[6,6])}-{LL[self._burn_in:].sum()}'), plt.tight_layout(), plt.show()
            if sum_loglike == True:
                ll = LL[self._burn_in:].sum()
                return np.ones((1)) * - np.inf if np.isnan(ll) else ll
            else:
                return LL[self._burn_in:]
        except:
            if sum_loglike == True:
                return np.ones((1)) * - np.inf
            else:
                return np.ones((self._N, 1))[self._burn_in:] * - np.inf

    def loglike_profile(self, kwargs_ll, sum_loglike=True):
        try:
            c, Phi, Q, H, betas = kwargs_ll
            kwargs_kf = (self.yx, self.builder._Zf, self.FIXEDx, betas, c, Phi, Q, H,
                         self.a1, self.P1, self._N, self._k, self.Sx, self.Ix, self.Sq, self.Iq, self.builder._jacobian)
            # a, P, a_t, P_t, LL2 = EKF(*kwargs_kf)
            # a, P, a_t, P_t, LL1 = CEKF(*kwargs_kf)
            try:
                a, P, a_t, P_t, LL = self._filter(*kwargs_kf)[:5]
            except:
                a, P, a_t, P_t, LL = self._filter(*kwargs_kf)[:5]
            # pd.DataFrame(a_t[:, :, 0]).plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
            # pd.DataFrame(a_t[:, 4:7, 0]).plot(figsize=(20, 10), title=f'{np.log(Q[6,6])}-{LL[self._burn_in:].sum()}'), plt.tight_layout(), plt.show()
            if sum_loglike == True:
                ll = LL[self._burn_in:].sum()
                return (np.ones((1)) * - np.inf, np.zeros((self._N, self._k, 1))) if np.isnan(ll) else (ll, a_t)
        except:
                return (np.ones((1)) * - np.inf, np.zeros((self._N, self._k, 1)))

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
        kwargs_ll, c, Phi, Q, H, betas, untransformed_params = self._build_kwargs_kf(transformed_params)
        sum_ll = self.loglike(kwargs_ll=kwargs_ll)
        # self.__print_params(untransformed_params)
        print(sum_ll)
        return sum_ll

    def objf_profile(self, transformed_params):
        kwargs_ll, c, Phi, Q, H, betas, untransformed_params = self._build_kwargs_kf(transformed_params)
        sum_ll, a_t = self.loglike_profile(kwargs_ll=kwargs_ll)
        # self.__print_params(untransformed_params)
        print(sum_ll)
        return sum_ll, a_t

    def estimate(self):
        self.config_mle()
        func = lambda params: - self.objf(params) / self.nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        start = time()
        self._mle_optim_results = minimize(fun=func, x0=self.mle_inits, method='SLSQP',
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
        kwargs_ll, c, Phi, Q, H, betas, untransformed_params = self._build_kwargs_kf(transformed_estimates)
        return (untransformed_params, c, Phi, Q, H, betas, self.a1, self.P1, self.collapse, self.unscented,
                self.builder._u_x, self.builder._v_tau, self._mle_optim_results, self._run_time, self.Sq, self.Iq)

