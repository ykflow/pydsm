import numpy as np
from monte_carlo.config_monte_carlo_experiment import ConfigMonteCarloExperiment
from monte_carlo.single_monte_carlo_experiment import SingleMonteCarloExperiment
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from joblib import Parallel, delayed, dump


class MonteCarloExperiments:
    def __init__(self, x_grid:np.array, tau_grid: np.array, N_fit:int=1000, N_for:int=1000, f_true:np.array=None):

        self.x_grid = x_grid
        self.tau_grid = tau_grid
        self.N_fit = N_fit
        self.N_for = N_for
        self.f_true = f_true
        if self.f_true is not None:
            self.N_fit = self.f_true.shape[0]
            self.N_for = 0

        self.N = self.N_fit + self.N_for


        self.m = self.x_grid.shape[0]
        self.n = self.tau_grid.shape[0]
        self._args_config = (self.x_grid, self.tau_grid, self.N, self.f_true)
        self.configurator = ConfigMonteCarloExperiment(*self._args_config)
        self._FLAG_CONFIG_COMPLETE = False


    def specify_experiment(self, fit:bool=False, mean:str='carr_wu_standard', variance='iid', missing_moneyness:bool=False, missing_rate: float=0.,
                           M: int=1, collapse:bool=True, unscented:bool=False, k_factors_linear:int=5,
                           x_pillars:np.array=None, tau_pillars:np.array=None, cyclical_knots:np.array=None):


        self._FLAG_FIT_MODEL = fit
        self.measurement = mean
        self.variance = variance
        self.missing_moneyness = missing_moneyness
        self.missing_rate = missing_rate
        self.M = M
        self.collapse = collapse
        self.unscented = unscented
        self.k_factors_linear = k_factors_linear
        self.x_pillars = x_pillars
        self.tau_pillars = tau_pillars
        self.cyclical_knots = cyclical_knots

        self._args_experiment_specification = (self._FLAG_FIT_MODEL, self.measurement, self.variance,
                                               self.missing_moneyness, self.missing_rate,
                                               self.M, self.collapse, self.unscented, self.k_factors_linear,
                                               self.x_pillars, self.tau_pillars, self.cyclical_knots)

        self.configurator.specify_experiment(*self._args_experiment_specification)
        self._FLAG_CONFIG_COMPLETE = self.configurator._FLAG_CONFIG_COMPLETE

    def _process_experiment_results(self):
        self._mc_results_summary = dict()
        self._mc_results_combined = dict()
        n_results = len(self._experiment_labels)
        if 'f_hat' in self._experiment_labels:
            n_results -= 1

        for i in range(n_results):
            label = self._experiment_labels[i]
            cols = self.configurator._k_params if label == 'mle' else self.configurator.k_states
            results_i = np.zeros((self.M, cols)) * np.nan

            for j in range(self.M):
                try:
                    results_i[j] = np.array(self._mc_results[j][i]).flatten()
                except:
                    pass

            results_summary_i = np.nanmean(results_i, axis=0).T
            self._mc_results_combined[self._experiment_labels[i]] = results_i
            self._mc_results_summary[self._experiment_labels[i]] = results_summary_i

        if 'f_hat' in self._experiment_labels:
            factors = np.zeros((self.M, self.N, self.configurator.k_states))
            for j in range(self.M):
                try:
                    factors[j] = self._mc_results[j][-1][:, :, 0]
                except:
                    pass

            self._mc_results_combined['f_hat'] = factors

        # mle_bias = self._mc_results_summary['mle'].copy()
        # if self.configurator._FLAG_SIMULATE:
        #     mle_bias -= self.configurator._true_sim_params
        #     mle_bias = np.concatenate((self.configurator._true_sim_params, mle_bias), axis=1)
        # else:
        #     true_params = mle_bias[:, 0].reshape(len(mle_bias), 1)
        #     mle_bias[:, 1:] -= true_params
        # self._mc_results_summary['mle_bias'] = mle_bias

    def perform_expirements(self, n_jobs=-3, ekf_and_cekf:bool=False, ukf_and_cekf:bool=False):
        if not self._FLAG_CONFIG_COMPLETE:
            raise ('Specify expirement first in function "specify_forecasting_experiment()"')

        self._n_jobs = n_jobs
        if not ekf_and_cekf and not ukf_and_cekf:
            unscented = False
            experiments = []
            for i in range(self.M):
                args = (self.configurator.list_df_surfaces[i], self.configurator._f,
                        self.configurator._args_measurement_eq, self.configurator._args_fixed_params,
                        self.configurator._FLAG_FIT_MODEL, self.configurator.collapse, unscented, self.N_fit, self.N_for, self.configurator.mle_inits)
                experiment_i = SingleMonteCarloExperiment(*args)
                experiments.append(experiment_i)
            self._mc_results = Parallel(n_jobs=n_jobs, verbose=3)(delayed(lambda x: x.run())(experiment) for experiment in experiments)
            self._experiment_labels = experiments[0]._experiment_labels
            self._process_experiment_results()

        elif ekf_and_cekf:
            unscented = False
            self.dict_mc_combined = dict()
            self.dict_mc_summary = dict()
            for collapsed in [False, True]:
                print(collapsed)
                experiments = []
                for i in range(self.M):
                    untransformed_params, c, Phi, Q, H, betas, a1, P1, _, _ = self.configurator._args_fixed_params
                    args_fixed_params = (untransformed_params, c, Phi, Q, H, betas, a1, P1, collapsed, False)
                    args = (self.configurator.list_df_surfaces[i],  self.configurator._f,
                            self.configurator._args_measurement_eq, args_fixed_params,
                            self.configurator._FLAG_FIT_MODEL, collapsed, unscented, self.N_fit, self.N_for, self.configurator.mle_inits)

                    experiment_i = SingleMonteCarloExperiment(*args)
                    experiments.append(experiment_i)
                self._mc_results = Parallel(n_jobs=n_jobs, verbose=3)(delayed(lambda x: x.run())(experiment) for experiment in experiments)

                self._experiment_labels = experiments[0]._experiment_labels
                self._process_experiment_results()
                self.dict_mc_combined[f'CEKF_{collapsed}_mc_results_summary'] = self._mc_results_combined
                self.dict_mc_summary[f'CEKF_{collapsed}_mc_results_summary'] = self._mc_results_summary

        elif ukf_and_cekf:
            self.dict_mc_summary = dict()
            self.dict_mc_combined = dict()
            for ukf in [False, True]:
                print(ukf)
                label = 'UKF' if ukf else 'CEKF'
                collapse = False if ukf else True
                experiments = []
                for i in range(self.M):
                    untransformed_params, c, Phi, Q, H, betas, a1, P1, _, _ = self.configurator._args_fixed_params
                    args_fixed_params = (untransformed_params, c, Phi, Q, H, betas, a1, P1, collapse, ukf)
                    args = (self.configurator.list_df_surfaces[i],  self.configurator._f,
                            self.configurator._args_measurement_eq, args_fixed_params,
                            self.configurator._FLAG_FIT_MODEL, collapse, ukf, self.N_fit, self.N_for, self.configurator.mle_inits)

                    experiment_i = SingleMonteCarloExperiment(*args)
                    experiments.append(experiment_i)
                self._mc_results = Parallel(n_jobs=n_jobs, verbose=3)(delayed(lambda x: x.run())(experiment) for experiment in experiments)

                self._experiment_labels = experiments[0]._experiment_labels
                self._process_experiment_results()
                self.dict_mc_combined[f'{label}_mc_results_summary'] = self._mc_results_combined
                self.dict_mc_summary[f'{label}_mc_results_summary'] = self._mc_results_summary

