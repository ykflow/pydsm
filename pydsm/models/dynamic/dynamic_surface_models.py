import numpy as np
from estimation.mle_dynamic_surface_models import MLESurfaceModels
from measurement_equations.measurement_design_builder import SurfaceMeasurementDesignBuilder
from scipy.optimize import OptimizeResult
from numba.typed import List
from utilities.special_numba_funcs import locate_missings
from datetime import datetime
import pandas as pd
from patsy.mgcv_cubic_splines import cc as cyclic_cubic_spine_builder
from scipy.interpolate import interp1d
from smoothers.kalman_smoother import collapsed_kalman_filter_smoother
from smoothers.simulation_smoother import sim_zero_mean_state, sim_nonlinear_measurement
from pandas.tseries.offsets import BDay


class DynamicSurfaceModels:
    def __init__(self, df_surface):
        self.df_surface = df_surface
        self._time_line = self.df_surface.index.unique()
        self._N = len(self._time_line)
        self._x_grid = np.sort(np.array([self.df_surface.CLOSEST_MONEYNESS.unique()]).astype(float)).T
        self._tau_grid = np.sort(np.array([self.df_surface.drop(columns='CLOSEST_MONEYNESS').columns.values]).astype(float).T)
        self._p = self._x_grid.shape[0] * self._tau_grid.shape[0]
        self._m, self._n = len(self._x_grid), len(self._tau_grid)
        self._PROFILE_FLAG_MLE_CONFIG_COMPLETE = False

    @staticmethod
    def vec(A):
        return A.reshape((-1, 1), order="F").astype(float)

    @staticmethod
    def unvec(A,m,n):
        return A.reshape(n,m).T

    @staticmethod
    def day_of_year(t: datetime):
        return ((t - datetime(t.year, 1, 1)).days + 1) / 365

    @staticmethod
    def ols_rsquared(y, y_hat):
        TSS = ((y - y.mean())**2).sum()
        RSS = ((y - y_hat)**2).sum()
        return 1 - RSS/TSS

    @staticmethod
    def _cvg(true, simulations, alpha:float=0.05):
        lb = np.nanquantile(simulations, alpha/2, axis=0)
        ub = np.nanquantile(simulations, 1-alpha / 2, axis=0)
        cvg = 1 - ((true > ub).sum() + (true < lb).sum()
                   ) / len(true)
        return cvg * 100

    @staticmethod
    def _jac(Zf, a, FIXED, betas, p, k):
        h = 1e-8 * (np.fabs(a) + 1e-8)  # Find stepsize
        h = np.maximum(h, 5e-6)

        Jac = np.zeros((p, k), dtype=np.float64)
        for i in range(k):
            h_i_min = a.copy()
            h_i_plus = a.copy()
            h_i_min[i] -= h[i]
            h_i_plus[i] += h[i]
            Jac[:, i] = ((Zf(h_i_plus, FIXED, betas, p, k) - Zf(h_i_min, FIXED, betas, p, k)) / (2 * h[i])).flatten()

        return Jac

    def _cyclical_spline_builder(self, tau:np.array):
        return cyclic_cubic_spine_builder(tau, knots=self.cyclical_knots, lower_bound=0, upper_bound=1)

    def prepare_fixed_input_ukf(self):
        zeros_y_p = np.zeros((self._p, 1), dtype=np.float64)
        eye_p = np.eye(self._p)

        self.B = eye_p
        self._k_cyclical_knots = None
        if self.cyclical_knots is not None:
            self.B = self._cyclical_spline_builder(self._tau_grid)
            self._k_cyclical_knots = self.B.shape[1]

        yx = List()
        FIXEDx = List()
        Ix = np.zeros(self._N)
        Sx = List()
        px = np.zeros(self._N)
        for t in range(self._N):
            time_point = self._time_line[t]
            tmp = self.df_surface.loc[[time_point]].sort_values(by='CLOSEST_MONEYNESS')
            Y = tmp.drop(columns='CLOSEST_MONEYNESS').copy().values.reshape(self._m, self._n)
            x = tmp.CLOSEST_MONEYNESS.values.reshape(self._m,1) * np.ones((self._m,self._n))
            tau = tmp.drop(columns='CLOSEST_MONEYNESS').columns.values.reshape(self._n, 1).T * np.ones((self._m, self._n))
            T = self.day_of_year(time_point) + tau

            vecY = self.vec(Y)
            vecx = self.vec(x)
            vectau = self.vec(tau)
            vecT = self.vec(T)

            if self.cyclical_knots is not None:
                vecB = self._cyclical_spline_builder(vectau)
            else:
                vecB = eye_p
            self.vecB = vecB.copy()

            loc_missing, nobs_missing, loc_complete, nobs_complete = locate_missings(np.copy(vecY))
            px[t] = nobs_complete
            mask = loc_complete.flatten()
            S = np.eye(self._p)[mask]
            y_tmp = np.copy(vecY)
            y_tmp[loc_missing.flatten()] = 0  # put NA to zero to enable matrix algebra

            # Create star (x) matrices
            if nobs_complete > 0:
                Ix[t] = 1.
                yx.append(S @ y_tmp)
                Sx.append(S)

                FIXED_tmp = List()
                FIXED_tmp.append(S @ vecx)
                FIXED_tmp.append(S @ vectau)
                FIXED_tmp.append(S @ vecT)
                FIXED_tmp.append(S @ self.Z[t]) if self._FLAG_LINEAR else FIXED_tmp.append(S @ vecB)
                FIXEDx.append(FIXED_tmp)

            else:
                Ix[t] = 0.
                yx.append(zeros_y_p)
                Sx.append(eye_p)

                FIXED_tmp = List()
                FIXED_tmp.append(vecx)
                FIXED_tmp.append(vectau)
                FIXED_tmp.append(vecT)
                FIXED_tmp.append(self.Z[t]) if self._FLAG_LINEAR else FIXED_tmp.append(vecB)
                FIXEDx.append(FIXED_tmp)

        return yx, px, FIXEDx, Sx, Ix


    def specify_measurement_equation(self, mean: str = 'carr_wu_standard', variance: str = 'iid',
                                     moneyness_pillars: np.array = None, maturity_pillars: np.array = None,
                                     Z: np.array = None,
                                     cyclical_knots:np.array=None, k_factors_linear:int=1, k_ar1_factors:int=0):
        self.Z = Z
        self._FLAG_LINEAR = True if self.Z is not None else False
        if self._FLAG_LINEAR:
            k_factors_linear = Z.shape[2]

        self._FLAG_MLE_COMPLETE = False

        if not self._FLAG_LINEAR and mean == 'linear':
            raise ValueError

        self._mean = mean
        self._variance = variance
        self._moneyness_pillars = moneyness_pillars
        self._maturity_pillars = maturity_pillars
        self.cyclical_knots = cyclical_knots
        self.k_factors_linear = k_factors_linear
        self.k_ar1_factors = k_ar1_factors

        self._k_cyclical_knots = None
        if self.cyclical_knots is not None:
            self.B = self._cyclical_spline_builder(self._tau_grid)
            self._k_cyclical_knots = self.B.shape[1]


        # self._yx, self._px, self._FIXEDx, self._Sx, self._Ix = self.prepare_fixed_input_ukf()
        self._builder = SurfaceMeasurementDesignBuilder(self._x_grid, self._tau_grid)
        self._builder.specify_measurement_equation(self._mean, self._variance, self._moneyness_pillars, self._maturity_pillars,
                                                   self.k_factors_linear, self.k_ar1_factors, self._k_cyclical_knots)
        self._yx, self._px, self._FIXEDx, self._Sx, self._Ix = self.prepare_fixed_input_ukf()

    def fit(self, collapse:bool=True, unscented:bool=False, mle_inits=None, N_fit:int=None,
            cross_sectional:bool=False, refit_cs:bool=False,
            find_filter_inits:bool=True, a1:np.array=None, P1:np.array=None,
            Sq:np.array=None, Iq:np.array=None):
        self._collapse = collapse
        self._unscented = unscented
        self._N_fit = N_fit if not None else self._N
        self._mle = MLESurfaceModels(self._yx[:self._N_fit], self._FIXEDx[:self._N_fit], self._Sx[:self._N_fit], self._Ix[:self._N_fit], self._builder,
                                     collapse=self._collapse, unscented=self._unscented, mle_inits=mle_inits,
                                     find_filter_inits=find_filter_inits,
                                     a1=a1, P1=P1, Sq=Sq, Iq=Iq)

        if cross_sectional:
            self.f_cs, self.beta_cs, self._rmspe_cs = self._mle.find_state_inits(fit_daily=True, refit=refit_cs)
            self._k = self._builder._k_factors
        else:
            self._mle.estimate()
            self._FLAG_MLE_COMPLETE = True
            self._set_model_params(*self._mle.gather_results())
            self._estimates = self._mle.get_untransformed_params()
            self._k = self._builder._k_factors
            self.run_filter()


    def _compute_standard_errors(self, N_fit:int, burn_in:int):
        if self._mle is None:
            raise('Fit model first!!!')

        n = N_fit - burn_in
        k = self._estimates.shape[0]
        # eps = self._estimates * 1e-5
        eps = 1e-8 * (np.fabs(self._estimates) + 1e-8)
        eps = np.maximum(eps, 5e-6)


        actual_ll = self.LL[burn_in:N_fit].flatten()

        scores = np.zeros((n, k), dtype=np.float64)
        for i in range(k):
            # print(f'progress={((i+1)/k)*100}%')
            delta = np.zeros((k,1))
            delta[i] += eps[i]

            params_plus_delta = self._estimates + delta
            params_minus_delta = self._estimates - delta

            kwargs_ll_plus = self._mle._build_kwargs_kf(self._mle.from_untransformed_params_to_transformed_params(params_plus_delta))[0]
            kwargs_ll_minus = self._mle._build_kwargs_kf(self._mle.from_untransformed_params_to_transformed_params(params_minus_delta))[0]

            plus_ll = self._mle.loglike(kwargs_ll=kwargs_ll_plus, sum_loglike=False).flatten()[burn_in:N_fit]
            minus_ll = self._mle.loglike(kwargs_ll=kwargs_ll_minus, sum_loglike=False).flatten()[burn_in:N_fit]

            counter_plus_ll = 0 if np.isinf(np.abs(plus_ll)).sum() > 0 else 1
            counter_minus_ll = 0 if np.isinf(np.abs(minus_ll)).sum() > 0 else 1
            counter_total = counter_plus_ll + counter_minus_ll
            h = np.inf if counter_total == 0 else counter_total
            ll_up = plus_ll if counter_plus_ll == 1 else actual_ll
            ll_down = minus_ll if counter_minus_ll == 1 else actual_ll

            scores[:, i] = (ll_up - ll_down) / (h * eps[i])

        opg = (scores.T @ scores)
        self._opg = (opg + opg.T) / 2  # ensure symmetry
        self._standard_errors = np.sqrt(np.diag(np.linalg.pinv(self._opg))).reshape(k,1)


    def profile(self, collapse:bool=True, unscented:bool=False, mle_inits=None, N_fit:int=None,
            cross_sectional:bool=False, refit_cs:bool=False,
            find_filter_inits:bool=True, a1:np.array=None, P1:np.array=None,
            Sq:np.array=None, Iq:np.array=None, transformed_params=None):

        if not self._PROFILE_FLAG_MLE_CONFIG_COMPLETE:
            self._collapse = collapse
            self._unscented = unscented
            self._N_fit = N_fit if not None else self._N
            self._mle = MLESurfaceModels(self._yx[:self._N_fit], self._FIXEDx[:self._N_fit], self._Sx[:self._N_fit], self._Ix[:self._N_fit], self._builder,
                                         collapse=self._collapse, unscented=self._unscented, mle_inits=mle_inits,
                                         find_filter_inits=find_filter_inits,
                                         a1=a1, P1=P1, Sq=Sq, Iq=Iq)
            self._mle.config_mle()

            self._PROFILE_FLAG_MLE_CONFIG_COMPLETE = True

        if transformed_params is not None:
            return self._mle.objf_profile(transformed_params)


    def set_profile_estimates(self):
        self._FLAG_MLE_COMPLETE = True
        self._set_model_params(*self._mle.gather_results())
        self._estimates = self._mle.get_untransformed_params()
        self._k = self._builder._k_factors
        self.run_filter()


    def _set_model_params(self, untransformed_params, c, Phi, Q, H, betas, a1, P1, collapse, unscented,
                          u_x:interp1d=None, v_tau:interp1d=None,
                          optim_results:OptimizeResult=None, run_time:float=None, Sq=None, Iq=None):
        self._estimates = untransformed_params
        self.c = c
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.betas = betas
        self.a1 = a1
        self.P1 = P1
        self._collapse = collapse
        self._unscented = unscented
        self._u_x = u_x
        self._v_tau = v_tau
        self._optim_results = optim_results
        self._run_time = run_time
        self.Sq = Sq
        self.Iq = Iq
        self._k = self._builder._k_factors
        self._k_betas = len(self.betas)
        self._k_params = len(self._estimates)

        if not self._FLAG_MLE_COMPLETE:
            self._mle = self._mle = MLESurfaceModels(self._yx, self._FIXEDx, self._Sx, self._Ix, self._builder,
                                                     collapse=self._collapse, unscented=self._unscented, find_filter_inits=False, Sq=Sq, Iq=Iq)
            self._mle.config_mle()
            self._mle.a1 = self.a1
            self._mle.P1 = self.P1
            self._mle._mle_optim_results = self._optim_results
            self._mle._run_time = self._run_time

    def _results_to_pickle(self):
        dict_results = dict()
        dict_results['set_params'] = [self._estimates, self.c, self.Phi, self.Q, self.H, self.betas,
                                      self.a1, self.P1, self._collapse, self._unscented, self._u_x, self._v_tau, self._optim_results, self._run_time]
        dict_results['param_dims'] = [self._k, self._k_betas, self._k_params]
        return dict_results

    def run_filter(self):
        if self._mle is None:
            raise ('Fit model first!!!')
        a, P, a_t, P_t, LL = self._mle._filter(self._yx, self._builder._Zf, self._FIXEDx, self.betas, self.c,
                                               self.Phi, self.Q, self.H, self.a1, self.P1,
                                               self._N, self._k, self._Sx, self._Ix, self._builder._jacobian)[:5]
        self.a = a
        self.P = P
        self.a_t = a_t
        self.P_t = P_t
        self.LL = LL


    def run_simulation_smoother(self, MC=100, horizon:int=252*2, vecX=np.array([[0], [0]]),vecTAU=np.array([[1/12], [3/12]])):
        if self._mle is None:
            raise ('Fit model first!!!')

        self.X = self._x_grid.reshape(self._m,1) * np.ones((self._m, self._n))
        self.TAU = self._tau_grid.reshape(self._n, 1).T * np.ones((self._m, self._n))
        self.vecX = self.vec(self.X)
        self.vecTAU = self.vec(self.TAU)
        self.vecB = self._cyclical_spline_builder(self.vecTAU)


        yx, FIXEDx, Sx, px, Ix, N = self._yx, self._FIXEDx, self._Sx, self._px, self._Ix, self._N
        timeline = self._time_line.tolist()
        if horizon > 0:
            for i in range(1, horizon+1):
                t_plus_i = self._time_line[-1] + BDay(i)
                timeline.append(t_plus_i)
                vecT = self.day_of_year(t_plus_i) + self.vecTAU
                FIXED = List()
                FIXED.append(self.vecX)
                FIXED.append(self.vecTAU)
                FIXED.append(vecT)
                FIXED.append(self.vecB)
                FIXEDx.append(FIXED)
                yx.append(np.zeros((self._p, 1), dtype=float))
                Sx.append(np.eye(self._p))

            px = np.concatenate((px, np.zeros(horizon, dtype=int)))
            Ix = np.concatenate((Ix, np.zeros(horizon, dtype=int)))
            N += horizon


        Isigma2 = np.ones(N)
        Isigma2[-horizon:] = 0
        simFIXED = List()
        for t in timeline:
            vecT = self.day_of_year(t) + vecTAU
            FIXED = List()
            FIXED.append(vecX.astype(float))
            FIXED.append(vecTAU.astype(float))
            FIXED.append(vecT.astype(float))
            FIXED.append(vecX.astype(float))
            simFIXED.append(FIXED)


        sigma2_eps_sim = self._u_x(vecX) * self._v_tau(vecTAU)

        _, _, _, _, _, y_star, H_star = self._mle._filter(yx, self._builder._Zf, FIXEDx, self.betas,
                                                          self.c, self.Phi, self.Q, self.H, self.a1, self.P1,
                                                          N, self._k, Sx, Ix, self._builder._jacobian)

        y_plus, a_plus = sim_zero_mean_state(self.Phi, self.Q, H_star, self.a1, self.P1, N, self._k, MC)
        p_sim = len(vecX)
        self.a_sim = np.zeros((N, self._k, MC))
        self.y_sim = np.zeros((N, p_sim, MC))
        self._time_line_sim = timeline
        for i in range(MC):
            a_plus_i = a_plus[:, :, i].reshape(N, self._k, 1)
            y_plus_i = y_plus[:, :, i].reshape(N, self._k, 1)
            y_tilde_i = y_star - y_plus_i
            a_hat_i = collapsed_kalman_filter_smoother(y_tilde_i, H_star, self.Phi, self.c, self.Q, self.a1, self.P1, Ix)[0]
            a_sim_i = a_plus_i + a_hat_i
            y_sim_i = sim_nonlinear_measurement(sigma2_eps_sim, self._builder._Zf, a_sim_i, simFIXED, self.betas, N, p_sim, self._k, Isigma2)
            self.a_sim[:, :, i] = a_sim_i.reshape(N, self._k)
            self.y_sim[:, :, i] = y_sim_i.reshape(N, p_sim)


    def compute_stats(self, N_fit:int=None):
        if N_fit is None:
            N_fit = self._N

        seasons = ['1', '2', '3', '4']

        k_s = 0
        if 'sknots' in  self._mean:
            k_s = self._k_cyclical_knots
        else:
            for s in seasons:
                if s in self._mean:
                    k_s = int(s) * 2

        self.ll = self.LL[self._mle._burn_in:].sum()
        nobs = self._mle.nobs
        self.aic = 2 * self._k_params - 2 * self.ll
        self.bic = self._k_params * np.log(nobs) - 2 * self.ll

        self.rmse = np.zeros(self._N)
        self.rmse_t = np.zeros(self._N)
        self.rmspe = np.zeros(self._N)
        self.cvg_95 = np.zeros(self._N)
        self.mae = np.zeros(self._N)
        self.mape = np.zeros(self._N)
        self.r2 = np.zeros(self._N)
        self.df_vols = self.df_surface.copy()
        self.df_vols.iloc[:, 1:] = self.df_vols.iloc[:, 1:] **.5
        self.df_fitted_vols = self.df_surface.copy()
        self.df_predicted_vols = self.df_surface.copy()
        self.df_fitted_deseasonalized_vols = self.df_surface.copy()
        self.df_residuals_vols = self.df_surface.copy()
        self.df_squared_residuals_vols = self.df_surface.copy()
        self.df_rel_squared_residuals_vols = self.df_surface.copy()
        self.df_absolute_residuals_vols = self.df_surface.copy()
        self.df_rel_absolute_residuals_vols = self.df_surface.copy()
        self.df_predicted_corr_in = pd.DataFrame(index=self._x_grid.flatten(), columns=self._tau_grid.flatten())
        self.df_predicted_corr_out = pd.DataFrame(index=self._x_grid.flatten(), columns=self._tau_grid.flatten())

        self.X = self._x_grid.reshape(self._m,1) * np.ones((self._m, self._n))
        self.TAU = self._tau_grid.reshape(self._n, 1).T * np.ones((self._m, self._n))

        self.vecX = self.vec(self.X)
        self.vecTAU = self.vec(self.TAU)
        self.vecB = self._cyclical_spline_builder(self.vecTAU)


        for t in range(self._N):
            ### DAILY AVG STATS
            yx = self._yx[t]**.5
            yx_hat = self._builder._Zf(self.a[t], self._FIXEDx[t], self.betas, self._px[t], self._k) ** .5
            yx_hat_t = self._builder._Zf(self.a_t[t], self._FIXEDx[t], self.betas, self._px[t], self._k) ** .5
            mse = ((yx - yx_hat_t) ** 2).mean()
            mse_t = ((yx - yx_hat_t) ** 2).mean()
            mspe = (((yx / yx_hat) - 1) ** 2).mean()
            mae = np.abs(yx - yx_hat_t).mean()
            mape = np.abs(yx / yx_hat_t - 1).mean()
            self.rmse[t] = np.sqrt(mse) * 100
            self.rmse_t[t] = np.sqrt(mse_t) * 100
            self.rmspe[t] = np.sqrt(mspe) * 100
            self.mae[t] = mae *100
            self.mape[t] = mape * 100
            self.r2[t] = self.ols_rsquared(yx.flatten(), yx_hat_t.flatten()) *100

            a_t = self.a[t].copy()  # filtered state
            P_t = self.P[t].copy()  # filtered state cov
            M = 500
            px = int(self._px[t])
            # sim_f = np.random.multivariate_normal(a_t.flatten(), P_t, size=M).reshape(M, self._k, 1)
            # sim_y = np.zeros((M, px, 1))
            not_na = np.invert(
                np.isnan(
                    self.vec(self.df_surface.loc[self._time_line[t], self._tau_grid.flatten()].values).flatten()
                )
            )
            # for j in range(M):
            #     tmp1 = self._builder._Zf(sim_f[j], self._FIXEDx[t], self.betas, px, self._k)
            #     tmp2 = tmp1 + np.random.normal(scale=np.diag(self.H) ** .5)[not_na].reshape(px, 1)
            #     sim_y[j] = tmp2
            # mu_y_hat = self._builder._Zf(self.a[t], self._FIXEDx[t], self.betas, px, self._k)
            # Jx = self._jac(self._builder._Zf, self.a[t], self._FIXEDx[t], self.betas, px, self._k)
            # cov_y_hat = Jx @ self.P[t] @ Jx.T + np.diag(np.diag(self.H)[not_na])
            # sim_y_hat = np.random.multivariate_normal(mu_y_hat.flatten(), np.diag(np.diag(self.H)[not_na]), size=M).reshape(M, px, 1)
            #
            # self.cvg_95[t] = self._cvg(yx, sim_y_hat ** .5, 0.05)

            #### FITTED SURFACE
            vecT = self.day_of_year(self._time_line[t]) + self.vecTAU
            FIXED = List()
            FIXED.append(self.vecX)
            FIXED.append(self.vecTAU)
            FIXED.append(vecT)
            FIXED.append(self.vecB)
            f = self.a_t[t].copy()
            fwos = self.a_t[t].copy()
            fwos[-k_s:] = 0

            Y = self.df_surface.loc[self._time_line[t], self._tau_grid.flatten()] ** .5
            y_hat = self._builder._Zf(self.a[t], FIXED, self.betas, self._px[t], self._k) ** .5
            y_hat_t = self._builder._Zf(f, FIXED, self.betas, self._px[t], self._k) ** .5
            y_hat_wos_t = self._builder._Zf(fwos, FIXED, self.betas, self._px[t], self._k) ** .5

            Y_hat = self.unvec(y_hat, self._m, self._n)
            Y_hat_t = self.unvec(y_hat_t, self._m, self._n)
            Y_hat_wos_t = self.unvec(y_hat_wos_t, self._m, self._n)

            self.df_predicted_vols.loc[self._time_line[t], self._tau_grid.flatten()] = Y_hat
            self.df_fitted_vols.loc[self._time_line[t], self._tau_grid.flatten()] = Y_hat_t
            self.df_fitted_deseasonalized_vols.loc[self._time_line[t], self._tau_grid.flatten()] = Y_hat_wos_t

            self.df_residuals_vols.loc[self._time_line[t], self._tau_grid.flatten()] = (Y - Y_hat_t)
            self.df_squared_residuals_vols.loc[self._time_line[t], self._tau_grid.flatten()] = (Y - Y_hat_t) ** 2
            self.df_rel_squared_residuals_vols.loc[self._time_line[t], self._tau_grid.flatten()] = (Y / Y_hat_t - 1) ** 2
            self.df_absolute_residuals_vols.loc[self._time_line[t], self._tau_grid.flatten()] = np.abs(Y - Y_hat_t)
            self.df_rel_absolute_residuals_vols.loc[self._time_line[t], self._tau_grid.flatten()] = np.abs(Y/Y_hat_t - 1)

        # 1+1

    def forecast(self, horizons:list=[1, 5, 21, 63], M:int=1000):
        self._dict_rmspe = dict()
        self._dict_rel_squared_residuals_vols = dict()

        for h in horizons:
            rmspe = np.zeros(self._N) * np.nan
            df_rel_squared_residuals_vols = self.df_surface.copy()
            df_rel_squared_residuals_vols.iloc[:, 1:] = df_rel_squared_residuals_vols.iloc[:, 1:] * np.nan
            for i in range(h, self._N):
                t_plus_h = self._time_line[i]
                vecT = self.day_of_year(t_plus_h) + self.vecTAU
                FIXED = List()
                FIXED.append(self.vecX)
                FIXED.append(self.vecTAU)
                FIXED.append(vecT)
                FIXED.append(self.vecB)

                Y = self.df_surface.loc[t_plus_h, self._tau_grid.flatten()].values ** .5
                y = self.vec(Y)
                not_na = np.invert(np.isnan(y))
                y = y[not_na]

                # a_t_h = self.a_t[i - h].copy()  # state forecast
                # F_t_h = self.P_t[i - h] + self.Q * h  # state forecast variance
                #
                # sim_f_t_h = np.random.multivariate_normal(a_t_h.flatten(), F_t_h, size=M).reshape(M, self._k, 1)
                # sim_y_t_h = np.zeros((M, self._p, 1))
                # for j in range(M):
                #     tmp1 = self._builder._Zf(sim_f_t_h[j], FIXED, self.betas, self._m * self._n, self._k)
                #     tmp2 = tmp1 + np.random.normal(scale=np.diag(self.H)**.5).reshape(self._p,1)
                #     sim_y_t_h[j] = tmp2
                #
                # sim_y_hat = np.nanmean(sim_y_t_h, axis=0)[not_na] **.5

                y_hat = self._builder._Zf(self.a_t[i - h], FIXED, self.betas, self._m * self._n, self._k) ** .5
                Y_hat = self.unvec(y_hat, self._m, self._n)
                y_hat = y_hat[not_na]

                mspe = (((y / y_hat) - 1) ** 2).mean()
                rmspe[i] = np.sqrt(mspe) * 100
                df_rel_squared_residuals_vols.loc[t_plus_h, self._tau_grid.flatten()] = (Y / Y_hat - 1) ** 2

            self._dict_rmspe[h] = rmspe
            self._dict_rel_squared_residuals_vols[h] = df_rel_squared_residuals_vols

