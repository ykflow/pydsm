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
from scipy.optimize import minimize, approx_fprime


class SeasonalSurfaceSVIModel:
    def __init__(self, df_surface:pd.DataFrame):
        self.df_surface = df_surface
        self._time_line = self.df_surface.index.unique()
        self._N = len(self._time_line)
        self._x_grid = np.sort(np.array([self.df_surface.CLOSEST_MONEYNESS.unique()]).astype(float)).T
        self._tau_grid = np.sort(np.array([self.df_surface.drop(columns='CLOSEST_MONEYNESS').columns.values]).astype(float).T)
        self._p = self._x_grid.shape[0] * self._tau_grid.shape[0]
        self._m, self._n = len(self._x_grid), len(self._tau_grid)
        self._PROFILE_FLAG_MLE_CONFIG_COMPLETE = False

        self.k_params = 3


    @staticmethod
    def vec(A):
        return A.reshape((-1, 1), order="F")

    @staticmethod
    def surface(x, theta, rho, gamma, eta):
        phi_theta = eta / ( (theta ** gamma) * ((1 + theta) ** (1 - gamma)) )
        phi_theta_x = phi_theta * x
        return theta * (1 + rho * phi_theta_x + ( (phi_theta_x + rho)**2 + (1-rho**2) )**.5) / 2

    @staticmethod
    def map_parameters(f):
        rho_star, gamma_star, eta_star = f.flatten()
        rho = (np.exp(rho_star) - np.exp(-rho_star)) / (np.exp(rho_star) + np.exp(-rho_star))
        gamma = 0.5 * (np.exp(gamma_star) / (1 + np.exp(gamma_star)))
        eta = np.exp(eta_star)
        return rho, gamma, eta


    def interpolate_atm_curves(self):
        tmp = self.df_surface[self.df_surface.CLOSEST_MONEYNESS == 0].iloc[:, 1:]
        tau = self._tau_grid.flatten()
        self.df_atm_interp = pd.DataFrame(index=self._time_line, columns=tau)
        for t in self._time_line:
            y = tmp.loc[t].values
            not_na = np.invert(np.isnan(y))
            self.df_atm_interp.loc[t] = interp1d(tau[not_na], y[not_na] * tau[not_na], bounds_error=False)(tau) / tau

        self.df_atm_interp = self.df_atm_interp.T.ffill().bfill().T


    def _objf_l2(self, x, theta, y, trans_params):
        rho, gamma, eta = self.map_parameters(trans_params)
        Y_hat = self.surface(x, theta, rho, gamma, eta)
        y_hat = self.vec(Y_hat)
        not_na = np.invert(np.isnan(y))
        eps = (y - y_hat)[not_na]
        sse = np.mean(eps.flatten()**2)

        if np.isnan(sse):
            return np.inf
        if eta * (1 + np.abs(rho)) > 2:
            return np.inf
        return sse


    def fit(self):
        self.interpolate_atm_curves()
        tau = self._tau_grid
        x = self._x_grid
        self.a_t = np.zeros((self._N, self.k_params))
        i = 0
        for t in self._time_line:
            print(t)
            Y = self.df_surface.loc[t].iloc[:, 1:].values * tau.T
            y = self.vec(Y)
            theta = self.df_atm_interp.loc[t].values.reshape(1, self._n) * tau.T

            func = lambda params: self._objf_l2(x, theta, y, params) *1000
            grad = lambda params: approx_fprime(params, func, 6.5e-6)
            np.random.seed(5)
            inits = np.random.normal(scale=0.1, size=self.k_params)
            optim = minimize(fun=func, x0=inits, method='SLSQP', jac=grad, options={'maxiter': 500})

            self.a_t[i, :] = np.array([self.map_parameters(optim.x)])

            i +=1

        self.a_t = pd.DataFrame(self.a_t).ffill().bfill().values


    @staticmethod
    def ols_rsquared(y, y_hat):
        TSS = ((y - y.mean())**2).sum()
        RSS = ((y - y_hat)**2).sum()
        return 1 - RSS/TSS

    def compute_stats(self):
        self.rmse = np.zeros(self._N) * np.nan
        self.rmspe = np.zeros(self._N) * np.nan
        self.mae = np.zeros(self._N) * np.nan
        self.mape = np.zeros(self._N) * np.nan
        self.r2 = np.zeros(self._N) * np.nan
        self.df_vols = self.df_surface.copy()
        self.df_vols.iloc[:, 1:] =  self.df_vols.iloc[:, 1:] **.5


        self.df_rel_squared_residuals_vols = self.df_surface.copy()
        self.df_rel_squared_residuals_vols.iloc[:, 1:]= self.df_rel_squared_residuals_vols .iloc[:, 1:] * np.nan
        self.df_residuals_vols = self.df_rel_squared_residuals_vols.copy()


        tau = self._tau_grid
        x = self._x_grid
        for h in [1]:
            for i in range(h, self._N):
                ### DAILY AVG STATS
                t_plus_h = self._time_line[i]
                t = self._time_line[i-h]
                Y = self.df_surface.loc[t_plus_h].iloc[:, 1:].values **.5
                y = self.vec(Y)
                not_na = np.invert(np.isnan(y))
                y = y[not_na]
                theta_min = self.df_atm_interp.loc[t].values.reshape(1, self._n) * tau.T
                Y_hat = (self.surface(x, theta_min, *self.a_t[i-h]) / tau.T) ** .5
                y_hat = self.vec(Y_hat)[not_na]

                mse = ((y - y_hat) ** 2).mean()
                mspe = (((y / y_hat) - 1) ** 2).mean()
                mae = np.abs(y - y_hat).mean()
                mape = np.abs((y / y_hat) - 1).mean()
                self.rmse[i] = np.sqrt(mse) * 100
                self.rmspe[i] = np.sqrt(mspe) * 100
                self.mae[i] = mae *100
                self.mape[i] = mape * 100
                self.r2[i] = self.ols_rsquared(y.flatten(), y_hat.flatten()) *100

                self.df_residuals_vols.loc[t_plus_h, self._tau_grid.flatten()] = Y - Y_hat
                self.df_rel_squared_residuals_vols.loc[t_plus_h, self._tau_grid.flatten()] = (Y / Y_hat - 1) ** 2


    def forecast(self, horizons:list=[1, 5, 21, 63]):
        self._dict_rmspe = dict()
        self._dict_rel_squared_residuals_vols = dict()

        tau = self._tau_grid
        x = self._x_grid
        for h in horizons:
            rmspe = np.zeros(self._N) * np.nan
            df_rel_squared_residuals_vols = self.df_surface.copy()
            df_rel_squared_residuals_vols.iloc[:, 1:] = df_rel_squared_residuals_vols.iloc[:, 1:] * np.nan
            for i in range(h, self._N):
                ### DAILY AVG STATS
                t_plus_h = self._time_line[i]
                t = self._time_line[i - h]
                Y = self.df_surface.loc[t_plus_h].iloc[:, 1:].values ** .5
                y = self.vec(Y)
                not_na = np.invert(np.isnan(y))
                y = y[not_na]
                theta_min = self.df_atm_interp.loc[t].values.reshape(1, self._n) * tau.T
                Y_hat = (self.surface(x, theta_min, *self.a_t[i - h]) / tau.T) ** .5
                y_hat = self.vec(Y_hat)[not_na]

                mspe = (((y / y_hat) - 1) ** 2).mean()
                rmspe[i] = np.sqrt(mspe) * 100
                df_rel_squared_residuals_vols.loc[t_plus_h, self._tau_grid.flatten()] = (Y / Y_hat - 1) ** 2

            self._dict_rmspe[h] = rmspe
            self._dict_rel_squared_residuals_vols[h] = df_rel_squared_residuals_vols

            # #### FITTED SURFACE
            # vecT = self.day_of_year(self._time_line[t]) + self.vecTAU
            # FIXED = List()
            # FIXED.append(self.vecX)
            # FIXED.append(self.vecTAU)
            # FIXED.append(vecT)
            # FIXED.append(self.vecB)
            # f = self.a_t[t].copy()
            # fwos = self.a_t[t].copy()
            # fwos[-k_s:] = 0
            #
            # Y = self.df_surface.loc[self._time_line[t], self._tau_grid.flatten()] ** .5
            # y_hat = self._builder._Zf(self.a[t], FIXED, self.betas, self._px[t], self._k) ** .5
            # y_hat_t = self._builder._Zf(f, FIXED, self.betas, self._px[t], self._k) ** .5
            # y_hat_wos_t = self._builder._Zf(fwos, FIXED, self.betas, self._px[t], self._k) ** .5
            #
            # Y_hat = self.unvec(y_hat, self._m, self._n)
            # Y_hat_t = self.unvec(y_hat_t, self._m, self._n)
            # Y_hat_wos_t = self.unvec(y_hat_wos_t, self._m, self._n)
            #
            #
            # self.df_fitted_vols.loc[self._time_line[t], self._tau_grid.flatten()] = Y_hat_t
            # self.df_fitted_deseasonalized_vols.loc[self._time_line[t], self._tau_grid.flatten()] = Y_hat_wos_t
            # self.df_rel_squared_residuals_vols.loc[self._time_line[t], self._tau_grid.flatten()] = (Y / Y_hat - 1) ** 2


    #
    #     # 1+1
    #
    #
    #
    #     # n, p, m = self.Z.shape
    #     # v = np.zeros((n, p, 1)) * np.nan
    #     # v_t = np.zeros((n, p, 1)) * np.nan
    #     # for t in range(n):
    #     #     v[t] = self.y[t] - self.Z[t] @ self.a[t] - self.Xbeta[t]
    #     #     v_t[t] = self.y[t] - self.Z[t] @ self.a_t[t] - self.Xbeta[t]
    #     #
    #     # self._v = v
    #     # self._v_t = v_t
    #






# rmspe_x = model.df_rel_squared_residuals_vols.reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS']).mean(axis=1) ** .5 * 100
# rmspe_x = rmspe_x.reset_index().pivot_table(index='DATE', columns='CLOSEST_MONEYNESS', values=0)
# rmspe_x.columns = pd.cut(np.abs(rmspe_x.columns), bins=[0, 0.05, 0.1, 0.2,0.3,  0.4], include_lowest=True)
# rmspe_x = rmspe_x.T.reset_index().groupby(by='index').mean().T
#
# df_errors = model.df_rel_squared_residuals_vols
# df_errors = df_errors[df_errors.index > train_end]
# df_rmspe = df_errors.groupby(by='CLOSEST_MONEYNESS').mean() **.5
# df_rmspe.index = pd.cut(df_rmspe.index, bins=[-0.4, -0.2, -0.05, 0.05, 0.2, 0.4], include_lowest=True)
# df_rmspe.columns = pd.cut(df_rmspe.columns,  bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
# df_rmspe = df_rmspe.reset_index().groupby(by='index').mean()
# df_rmspe = df_rmspe.T.reset_index().groupby(by='index').mean().T *HUNDRED
# df_rmspe['AVG'] = df_rmspe.mean(axis=1)
# df_rmspe.loc['AVG'] = df_rmspe.mean(axis=0)