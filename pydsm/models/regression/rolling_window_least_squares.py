import numpy as np
import pandas as pd
from models.regression.least_squares import OLS


class RollingWindowLeastSquareAnalysis:
    def __init__(self, y: pd.DataFrame, X_fxd: pd.DataFrame, X_term: dict=None):
        self.y = y
        self.X = X_fxd
        self.X_term = X_term
        self.n = self.y.shape[0]
        self.p = self.y.shape[1]
        self.k_fxd = self.X.shape[1]
        self.k_term = 0 if self.X_term is None else len(self.X_term.keys())
        self.k = self.k_fxd + self.k_term
        self.variable_names = ['Const'] + self.X.columns.tolist()
        if self.X_term is not None:
            self.variable_names += list(self.X_term.keys())

    @staticmethod
    def standardize(a: np.array, standardize:bool=False):
        _, m = a.shape
        if standardize:
            a_avg, a_sd = a.mean(axis=0).reshape(1,m), a.var(axis=0).reshape(1,m) ** .5
            a_strd = (a.copy() - a_avg) / a_sd
            return a_strd, a_avg, a_sd
        else:
            return a, np.zeros((1,m)), np.ones((1,m))

    @staticmethod
    def add_intercept(X: np.array):
        n, m = X.shape
        return np.concatenate((np.ones((n,1)), X), axis=1)

    def period_regression(self, frequency, standardize_X):
        dates = self.y.index.sort_values()
        periods = self.y.index.to_period(frequency).sort_values()
        eop = [dates[periods.isin([period])][-1] for period in periods.unique()]

        self.time_line = eop
        self.full_X = pd.DataFrame(columns=self.variable_names, index=self.time_line)
        self.chow_test_input = pd.DataFrame(columns=['n', 'k', 'SSR'], index=self.time_line)
        self.betas = self.full_X.copy()
        self.betas_hac_se = self.full_X.copy()
        self.r2_adj_and_contributions = self.full_X.copy()
        self.r2_adj_and_contributions.rename(columns={'Const': 'All'}, inplace=True)

        t = 0
        for period in periods.unique():
            try:
                y_t = self.y[self.y.index.to_period(frequency).isin([period])]
                X_fxd_t = self.X[self.X.index.to_period(frequency).isin([period])]
                X_term_t = list()
                if self.X_term is not None:
                    for key in self.X_term.keys():
                        X_term_i = self.X_term[key]
                        X_term_t.append(X_term_i[X_term_i.index.to_period(frequency).isin([period])])

                n = X_fxd_t.shape[0]
                y_t = y_t.values.reshape(n * self.p, 1, order='F').copy()
                X_fxd_t = X_fxd_t.values.reshape(n, self.k_fxd).copy()

                if standardize_X:
                    X_fxd_t -= X_fxd_t.mean(axis=0)
                    X_fxd_t /= X_fxd_t.std(axis=0)
                    if self.X_term is not None:
                        for i in range(self.k_term):
                            X_term_t[i] -= X_term_t[i].mean(axis=0)
                            X_term_t[i] /= X_term_t[i].std(axis=0)

                X_fxd_t = self.add_intercept(X_fxd_t)
                X_t = np.kron(np.ones((self.p,1)), X_fxd_t)
                if self.X_term is not None:
                    for i in range(self.k_term):
                        X_t = np.concatenate((X_t, X_term_t[i].values.reshape(n * self.p, 1, order='F').copy()), axis=1)

                n, k_act = X_t.shape
                model_t = OLS(y_t, X_t)
                model_t.fit()

                self.betas.iloc[t, :] = model_t._beta.flatten()
                self.betas_hac_se.iloc[t, :] = model_t._se.flatten()
                self.r2_adj_and_contributions.iloc[t, 0] = model_t._adj_r2
                self.chow_test_input.iloc[t, :] = np.array([n, k_act, model_t._RSS])

                for i in range(1, self.k + 1):
                    X_t_i = np.delete(X_t, i, axis=1)
                    model_t_i = OLS(y_t, X_t_i)
                    model_t_i.fit()
                    self.r2_adj_and_contributions.iloc[t, i] = model_t._adj_r2 - model_t_i._adj_r2
            except:
                pass

            t += 1

    def rolling_regression(self, horizon:int=252, standardize_X:bool=True, control_periods: pd.Series=None, control_variables:pd.DataFrame=None, signs:pd.DataFrame=None):
        self.time_line = self.y.index
        self.full_X = pd.DataFrame(columns=self.variable_names, index=self.time_line)
        self.betas = self.full_X.copy()
        self.betas_hac_se = self.full_X.copy()
        self.r2_adj_and_contributions = self.full_X.copy()
        self.r2_adj_and_contributions.rename(columns={'Const': 'All'}, inplace=True)
        self.chow_test_input = pd.DataFrame(columns=['n', 'k', 'SSR'], index=self.time_line)
        self.h = horizon
        self.signs = signs
        skip_dates = True if control_periods is not None else False
        add_control = True if control_variables is not None else False
        add_sign_adj = True if self.signs is not None else False

        for t in range(1, self.n - self.h):
            print(t)
            y_t = self.y.iloc[t:t + self.h]
            # y_t = y_t.clip(y_t.quantile(0.02, axis=0).values, y_t.quantile(0.98, axis=0).values).copy()
            X_fxd_t = self.X.iloc[t:t + self.h]
            # X_fxd_t = X_fxd_t.clip(X_fxd_t.quantile(0.02, axis=0).values, X_fxd_t.quantile(0.98, axis=0).values).copy()
            X_term_t = list()
            if self.X_term is not None:
                for key in self.X_term.keys():
                    X_term_tmp = self.X_term[key].iloc[t:t + self.h]
                    # X_term_tmp = X_term_tmp.clip(X_term_tmp.quantile(0.02, axis=0).values, X_term_tmp.quantile(0.98, axis=0).values).copy()
                    X_term_t.append(X_term_tmp)

            idx_isin_cntrl_prds = True if (skip_dates and y_t.index.isin(control_periods).sum() > 0) else False
            if skip_dates and idx_isin_cntrl_prds:
                y_t = y_t[~y_t.index.isin(control_periods)]
                X_fxd_t = X_fxd_t[~X_fxd_t.index.isin(control_periods)]
                if self.X_term is not None:
                    for key in self.X_term.keys():
                        X_term_i = self.X_term[key]
                        X_term_t[key] = X_term_i[~X_term_i.index.isin(control_periods)]

                n = y_t.shape[0]
            else:
                n = self.h

            y_t = y_t.values.reshape(n * self.p, 1, order='F').copy()
            X_fxd_t = X_fxd_t.values.reshape(n, self.k_fxd).copy()

            if standardize_X:
                X_fxd_t -= X_fxd_t.mean(axis=0)
                X_fxd_t /= X_fxd_t.std(axis=0)
                if self.X_term is not None:
                    for i in range(self.k_term):
                        X_term_t[i] -= X_term_t[i].mean(axis=0)
                        X_term_t[i] /= X_term_t[i].std(axis=0)

            if add_control and not idx_isin_cntrl_prds:
                It = control_variables.iloc[t:t + self.h]
                loc = It.sum().values > 0
                add_control = True if loc.sum() > 0 else False
                It = It.iloc[:, loc]
                X_fxd_t = np.concatenate((X_fxd_t, It), axis=1)

            X_fxd_t = self.add_intercept(X_fxd_t)
            X_t = np.kron(np.ones((self.p, 1)), X_fxd_t)
            if self.X_term is not None:
                for i in range(self.k_term):
                    X_t = np.concatenate((X_t, X_term_t[i].values.reshape(n * self.p, 1, order='F').copy()), axis=1)

            if add_sign_adj:
                sign_t_minus_1 = self.signs.iloc[t - 1:t - 1 + self.h]
                n = sign_t_minus_1.shape[0]
                sign_t_minus_1 = sign_t_minus_1.values.reshape(n * self.p, 1, order='F').copy()
                X_t[:, 1:] *= sign_t_minus_1

            n, k_act = X_t.shape
            model_t = OLS(y_t, X_t)
            model_t.fit()

            self.betas.iloc[t + self.h, :] = model_t._beta.flatten()
            self.betas_hac_se.iloc[t + self.h, :] = model_t._se.flatten()
            self.r2_adj_and_contributions.iloc[t + self.h, 0] = model_t._adj_r2
            self.chow_test_input.iloc[t + self.h, :] = np.array([n, k_act, model_t._RSS])

            for i in range(1, k_act):
                X_t_i = np.delete(X_t, i, axis=1)
                model_t_i = OLS(y_t, X_t_i)
                model_t_i.fit()
                self.r2_adj_and_contributions.iloc[t + self.h, i] = model_t._adj_r2 - model_t_i._adj_r2

    def fit(self, rolling=True, horizon:int=252, period='Q', standardize_X:bool=True, control_periods: pd.Series=None,control_variables:pd.DataFrame=None, sign:pd.DataFrame=None):
        if rolling:
            self.rolling_regression(horizon, standardize_X, control_periods, control_variables, sign)
        else:
            self.period_regression(period, standardize_X)





# import matplotlib.pyplot as plt
#
# n = 1000
# m = 4
# np.random.seed(5)
# X = np.concatenate((np.ones((n,1)), np.random.normal(size=(n,m-1), loc=np.arange(1, m), scale=np.arange(1, m))), axis=1)
# betas = np.arange(1, m+1).reshape(m,1)
# np.random.seed(7)
# y = X @ betas + np.random.normal(size=(n,1), scale=5)
#
# rwlsa = RollingWindowLeastSquareAnalysis(pd.DataFrame(y), pd.DataFrame(X).iloc[:, 1:])
# rwlsa.fit(horizon=252, standardize=False)
# rwlsa.betas.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
# rwlsa.betas_hac_se.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
# rwlsa.r2_adj_and_contributions.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
