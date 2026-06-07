import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from measurement_equations.measurement_design_builder import SurfaceMeasurementDesignBuilder
from numba.typed import List
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ConfigMonteCarloExperiment:
    def __init__(self, x_grid: np.array, tau_grid: np.array, N:int=None, f_true:np.array=None):
        self.x_grid = x_grid
        self.tau_grid = tau_grid
        self.N = N
        self.f_true = f_true
        self._FLAG_SKIP_SIM = False
        if self.f_true is not None:
            self._FLAG_SKIP_SIM = True
            self.N = self.f_true.shape[0]
            self.f0 = self.f_true[0]
            # self.f_var = pd.DataFrame(self.f_true[:, :, 0]).diff().var().values.reshape(self.f_true.shape[1],1) /2

            f_hat = pd.DataFrame(self.f_true[:, :, 0])
            Q1 = f_hat.quantile(0.25)
            Q3 = f_hat.quantile(0.75)
            IQR = Q3 - Q1
            f_hat = f_hat[~((f_hat < (Q1 - 1.5 * IQR)) |(f_hat > (Q3 + 1.5 * IQR))).any(axis=1)].reset_index(drop=True)
            self.f_var = f_hat.diff().var().values.reshape(self.f_true.shape[1], 1)

        self.m = self.x_grid.shape[0]
        self.n = self.tau_grid.shape[0]
        self.p = self.m * self.n

        self._days = 365
        self._business_days = 250
        self._numb_years = int((np.ceil(self.N / self._business_days) + 1))
        self._t_end = str(pd.Timestamp.today().strftime(format='%Y/%m/%d'))
        self._t_start = str((pd.Timestamp.today() - timedelta(days=self._numb_years*self._days)).strftime(format='%Y/%m/%d'))
        self._timeline = pd.date_range(start=self._t_start, end=self._t_end)
        self._timeline = self._timeline[self._timeline.dayofweek < 5][-self.N:]  # working days only
        self._cols = ['CLOSEST_MONEYNESS'] + self.tau_grid.flatten().tolist()
        self._df_surface = pd.DataFrame(index=np.sort(self._timeline.tolist() * self.m), columns=self._cols)
        self._df_surface['CLOSEST_MONEYNESS'] = np.repeat(self.x_grid.T, self.N, 0).flatten()

        self.Xmat = np.ones((self.m, self.n)) * self.x_grid
        self.TAUmat = np.ones((self.m, self.n)) * self.tau_grid.T

        self._burn_in = 500
        self._FLAG_CONFIG_COMPLETE = False


    @staticmethod
    def vec(A):
        return A.reshape((-1, 1), order="F").astype(float)

    @staticmethod
    def unvec(A, n, m):
        return A.reshape((n, m), order="F").astype(float)

    @staticmethod
    def day_of_year(t: datetime):
        return ((t - datetime(t.year, 1, 1)).days + 1) / 365

    @staticmethod
    def sim_states(c, Phi, Q, k, N, burn_in:int=500):
        N_ = N + burn_in
        # np.random.seed(7)
        eta = np.random.multivariate_normal(np.zeros(k), Q, size=(N_, )).reshape(N_, k, 1)
        f = np.zeros((N_, k, 1))
        f[0] = eta[0].copy()
        for t in range(N_-1):
            f[t+1] = c + Phi @ f[t] + eta[t]
        return f[burn_in:]


    def _set_true_measurement_noise_params(self):
        if self.variance == 'iid':
            self._H_variances = np.ones((1,1)) * 0.005

        elif self.variance == 'diag':
            self._H_variances = np.ones((self.p, 1)) * 0.01
            if self.mean == 'linear' or self.mean == 'exponential':
                self._H_variances = np.arange(1, 7)[::-1][:self.p].reshape(self.p,1) *0.025/10

        elif self.variance == 'mvn-spline':
            self._p_x = self.x_pillars.shape[0] - 1
            self._p_tau = self.tau_pillars.shape[0]
            self._p_pillars = self._p_x + self._p_tau
            self._idx_x = self.x_pillars[:, 0] != 0
            self._H_variances = np.concatenate((self.x_pillars[self._idx_x, 1],
                                               self.tau_pillars[:, 1])).reshape(self._p_pillars, 1)
        else:
            raise ValueError


    def _set_true_state_params_linear(self):
        if self.mean == 'linear' or self.mean == 'exponential':
            self.c = np.zeros((self.k_states, 1))
            self.Phi = np.eye(self.k_states) * 0.95
            self.Q = np.eye(self.k_states) * 0.01
            self.betas = np.array([])
        else:
            # np.random.seed(333)
            self.c = np.zeros((self.k_states, 1)) - 0.025
            self.Phi = np.eye(self.k_states) * 0.97
            self.Q = np.eye(self.k_states) * 0.01
            self.betas = np.array([])

        self.f0 = np.linalg.inv(np.eye(self.k_states) - self.Phi) @ self.c

    def _config_state_space(self):

        self._builder = SurfaceMeasurementDesignBuilder(self.x_grid, self.tau_grid)
        self._builder.specify_measurement_equation(self.mean, self.variance,
                                                   self.x_pillars[:, 0], self.tau_pillars[:, 0],
                                                   k_factors_linear=self.k_factors_linear)

        self.k_states = self._builder._k_factors  # num factors

        if not self._FLAG_SKIP_SIM:
            self._set_true_state_params_linear()
        self._set_true_measurement_noise_params()

        try:
            self.Z = np.kron(np.ones((self.N, 1)), np.eye(self.p)).reshape(self.N, self.p, self.k_states)
        except:
            # np.random.seed(7)
            Z = np.abs(np.random.normal(size=(self.p, self.k_states)))
            Z /= Z.sum(axis=1).reshape(self.p, 1)
            self.Z = np.kron(np.ones((self.N, 1)), Z).reshape(self.N, self.p, self.k_states)

        self._act_Z = self.Z if self.mean == 'linear' or self.mean == 'exponential' else None

        self.H = self._builder._build_covH(self._H_variances)
        self.diagH = np.diag(self.H).reshape(self.p,1)

        if not self._FLAG_SKIP_SIM:
            self._params = np.concatenate((self.c.flatten(), np.diag(self.Phi),
                                           np.diag(self.Q), self._H_variances.flatten()), axis=0)
            self._k_params = self._params.shape[0]
            self._params = self._params.reshape(self._k_params, 1)
            self.untransformed_params = np.concatenate((self.c.flatten(),
                                                        -np.log((1 - np.diag(self.Phi)) / np.diag(self.Phi)),
                                                        np.log(np.diag(self.Q)),
                                                        np.log(self._H_variances.flatten())), axis=0).reshape(self._k_params, 1)
            self.mle_inits = np.random.normal(self.untransformed_params * 0.99, scale=0.01)
            self.a1 = self.f0.copy()
            self.P1 = self.Q.copy() * 10
        else:
            k_tmp = self.f_true.shape[1]
            self.c = np.zeros((k_tmp,1))
            self.Phi = np.eye(k_tmp)
            self.Q = np.diag(self.f_var.flatten() + 1e-6)
            self.betas = np.array([])
            self._params = np.array([])
            self.untransformed_params = np.array([])
            self.mle_inits = np.concatenate((self.f_var.flatten(),  self._H_variances.flatten()*1.5), axis=0)
            self._k_params = self.mle_inits.shape[0]
            self.mle_inits = np.log(self.mle_inits.reshape(self._k_params,1) + 1e-6)

            self.a1 = self.f0.copy()
            self.P1 = np.diag(self.f_var.flatten() *2)

    def _missing_along_moneyness(self):
        loc = np.empty((self.m, self.n))
        for i in range(self.m):
            rate = np.abs(self.x_grid[i])
            loc[i, :] = np.random.binomial(n=1, p=rate, size=self.n)
        return loc.astype(bool)

    def specify_experiment(self, fit:bool=False, mean:str='carr_wu_standard', variance='iid',
                           missing_moneyness:bool=False, missing_rate: float=0.25,
                           M: int=1, collapse:bool=True, unscented:bool=False, k_factors_linear:int=1,
                           x_pillars:np.array=None, tau_pillars:np.array=None, cyclical_knots:np.array=None):
        self._FLAG_FIT_MODEL = fit
        self.mean = mean
        # self._mean_for_sim = self.mean if not self._FLAG_SKIP_SIM else 'model_for_mc'
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

        self._config_state_space()

        self.list_df_surfaces = []
        for i in range(self.M):
            df_surface = self._df_surface.copy()
            Y_surface = np.zeros((self.N, self.m, self.n))

            if self._FLAG_SKIP_SIM:
                self._f = self.f_true.copy()
            else:
                self._f = self.sim_states(self.c, self.Phi, self.Q, self.k_states, self.N, self._burn_in)

            for t in range(self.N):
                time_point = self._timeline[t]
                T = self.day_of_year(time_point) + self.TAUmat
                FIXED = List()
                FIXED.append(self.Xmat)
                FIXED.append(self.TAUmat)
                FIXED.append(T)
                FIXED.append(self.Z[t])

                mu = self._builder._Zf(self._f[t], FIXED, self.betas, self.p, self.k_states)
                eps = (self.diagH**.5) * np.random.normal(size=(self.p, 1))
                loc = self._missing_along_moneyness() if self.missing_moneyness else np.random.binomial(n=1, p=self.missing_rate, size=(self.m, self.n)).astype(bool)
                Y = self.unvec(mu, self.m, self.n) + self.unvec(eps, self.m, self.n)
                Y[loc] = np.nan
                Y_surface[t] = Y.copy()
            df_surface.iloc[:, 1:] = Y_surface.reshape(self.N * self.m, self.n)
            self.list_df_surfaces.append(df_surface)

        self._FLAG_CONFIG_COMPLETE = True
        self._args_fixed_params = (self.untransformed_params, self.c, self.Phi, self.Q, self.H, self.betas, self.a1, self.P1, self.collapse, self.unscented)
        self._args_measurement_eq = (self.mean, self.variance, self.x_pillars[:, 0], self.tau_pillars[:, 0],
                                     self._act_Z, self.cyclical_knots, self.k_factors_linear)


# config = ConfigMonteCarloExperiment(1000, 12, 7)
# config.specify_experiment(missing_rate=0.0, M=500)