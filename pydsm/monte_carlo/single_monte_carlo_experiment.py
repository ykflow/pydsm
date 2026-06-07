import numpy as np
from scipy.stats import norm
from random import sample
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt


class SingleMonteCarloExperiment:
    def __init__(self, df_surface:pd.DataFrame, f_true: np.array, args_msrmnt_eq:tuple, args_fxd_pars:tuple=None,
                 flag_fit_model:bool=False, collapse:bool=True, unscented:bool=False, N_fit:int=None, N_for:int=None, inits=None):

        self.df_surface = df_surface
        self.f_true = f_true
        self.args_msrmnt_eq = args_msrmnt_eq
        self.args_fxd_pars = args_fxd_pars
        self._FLAG_FIT_MODEL = flag_fit_model
        self._collapse = collapse
        self._unscented = unscented
        self._N_fit = N_fit
        self._N_for = N_for
        self._inits = inits

        self._MC = 500
        self._alpha = 0.05
        self._tol_mle = 1e-7
        self._experiment_labels = []
        self._experiment_labels += ['mle'] if self._FLAG_FIT_MODEL else []
        self._experiment_labels += ["mse_f", "mae_f", "me_f", "cvg95%_f"]
        self._experiment_labels += ['f_hat'] if self._FLAG_FIT_MODEL else []


    @staticmethod
    def __compute_mae(true: np.array, forecast: np.array, exclude):
        mae = np.nanmean(np.abs(true - forecast), axis=0)  # NOTE true series may contain NaN
        mae[exclude] = np.nan
        return mae

    @staticmethod
    def __compute_corr(x: np.array, y: np.array):
        mu_x = x.mean(axis=0)
        mu_y = y.mean(axis=0)
        var_x = x.var(axis=0)
        var_y = y.var(axis=0)

        cov_xy = ((x - mu_x) * (y - mu_y)).mean(axis=0)
        rho_xy = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))
        return rho_xy

    @staticmethod
    def __compute_var(loc: np.array, scale: np.array, alpha: float):
        return loc + scale * norm.ppf(alpha)

    @staticmethod
    def __compute_es(loc: np.array, scale: np.array, alpha: float):
        return loc - (scale / alpha) * norm.pdf(norm.ppf(alpha))


    @staticmethod
    def _compute_coverage(true: np.array, lb, ub: np.array):
        n, m = true.shape
        points_below = (lb > true).sum(axis=0)
        points_above = (ub < true).sum(axis=0)
        coverage = 1 - (points_below + points_above) / n
        return coverage.reshape(m,1) *100

    def _perform_experiment(self):
        ### GATHER RESULTS
        results = []
        model = DynamicSurfaceModels(self.df_surface)
        model.specify_measurement_equation(*self.args_msrmnt_eq)
        if self._FLAG_FIT_MODEL:
            model.fit(collapse=self._collapse, unscented=self._unscented, N_fit=self._N_fit, mle_inits=self._inits)
            results.append(model._estimates)
        else:
            model._set_model_params(*self.args_fxd_pars)
            model.a1 = self.f_true[0]*0.98

        try:
            model.run_filter()
        except:
            model.run_filter()

        Pkk = model.P.diagonal(axis1=1, axis2=2)**.5
        lb, ub = model.a[:, :, 0] - 2 * Pkk, model.a[:, :, 0] + 2 * Pkk
        cvg95 = self._compute_coverage(self.f_true[:, :, 0], lb, ub)


        if self._N_for == 0:
            error = (self.f_true[:, :model.a.shape[1]] - model.a)
        else:
            error = (self.f_true[:, :model.a.shape[1]] - model.a)[self._N_fit:]

        fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(20,10), sharex=True)
        axs = ax.flatten()
        for i in range(model._k):
            axs[i].plot(model._time_line, self.f_true[:, i])
            axs[i].plot(model._time_line, model.a[:, i])

        plt.tight_layout()
        plt.show()

        # pd.DataFrame(model.LL).plot(figsize=(20,10)), plt.tight_layout, plt.show()

        me_f = error.mean(axis=0)
        mae_f = np.abs(error).mean(axis=0)
        mse_f = (error**2).mean(axis=0)
        results += [mse_f, mae_f, me_f, cvg95]

        if self._FLAG_FIT_MODEL:
            results.append(model.a)

        return results


    def run(self):
        try:
            results = self._perform_experiment()
            print("SUCCESS")
            return results
        except:
            return []

        # results = self._perform_experiment()