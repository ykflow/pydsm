import numpy as np
from scipy.optimize import minimize, approx_fprime
from datetime import datetime





class SeasonalSviPowerSurface:
    def __init__(self, t, tau: np.array, x: np.array, y: np.array):
        self.t = t
        self.y = y
        self.x = x
        self.tau = tau
        self.n, self.p = self.y.shape
        self._t = self.day_of_year(self.t)

        self.svi_k = 5
        self.curve_k = 3
        self._k_svi_params = self.svi_k * self.curve_k + 4

        self.dgr_u2 = 1
        self.dgr_v2 = 1

        self._idx_a = np.arange(self.curve_k)
        self._idx_b = np.arange(self.curve_k, 2*self.curve_k)
        self._idx_rho = np.arange(2*self.curve_k, 3*self.curve_k)
        self._idx_m = np.arange(3*self.curve_k, 4*self.curve_k)
        self._idx_s2 = np.arange(4*self.curve_k, 5*self.curve_k)
        self._idx_alphas = np.arange(5*self.curve_k, self._k_svi_params)

        self._idx_u2 = np.arange(self._k_svi_params, self._k_svi_params + self.dgr_u2 + 1)
        self._idx_v2 = np.arange(self._k_svi_params + self.dgr_u2 + 1, self._k_svi_params + self.dgr_u2 + self.dgr_v2 + 1)

        self.loc_notna = np.invert(np.isnan(self.vec(self.y))).flatten()
        self._nobs = self.loc_notna.sum()
        self._S = np.eye(self.n*self.p)[self.loc_notna]

        self._tau_pillars = np.unique(self.tau)
        self._x_pillars = np.unique(self.x)


    @staticmethod
    def day_of_year(t: datetime):
        return ((t - datetime(t.year, 1, 1)).days + 1) / 365

    @staticmethod
    def vec(A):
        return A.reshape((-1, 1), order="F")

    @staticmethod
    def _polynomial(tau, betas, d):
        result = np.zeros_like(tau)
        for i in range(d+1):
            result += betas[i] * tau**i
        return result


    def surface(self, tau, x, a, b, rho, m, s2, alpha):
        tmp = x - m
        T = self._t + tau
        p1 = 2 * np.pi * T
        p2 = 4 * np.pi * T
        theta = alpha[0] * np.sin(p1) + alpha[1] * np.cos(p1) + alpha[2] * np.sin(p2) + alpha[3] * np.cos(p2)
        return a + b * (rho * tmp + (tmp ** 2 + s2) ** .5) + theta *tau

    def surface_deseasoned(self, tau, x, a, b, rho, m, s2, alpha):
        tmp = x - m
        T = self._t + tau
        p1 = 2 * np.pi * T
        p2 = 4 * np.pi * T
        theta = alpha[0] * np.sin(p1) + alpha[1] * np.cos(p1) + alpha[2] * np.sin(p2) + alpha[3] * np.cos(p2)
        return a + b * (rho * tmp + (tmp ** 2 + s2) ** .5)

    def seasons(self, tau, x, a, b, rho, m, s2, alpha):
        T = self._t + tau
        p1 = 2 * np.pi * T
        p2 = 4 * np.pi * T
        theta = alpha[0] * np.sin(p1) + alpha[1] * np.cos(p1) + alpha[2] * np.sin(p2) + alpha[3] * np.cos(p2)
        return theta*b



    def _get_inits_mle(self):
        func = lambda params: self._objf_l2(params) / self._nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        # np.random.seed(5)
        inits = np.random.normal(scale=0., size=self._k_svi_params)
        optim = minimize(fun=func, x0=inits, method='L-BFGS-B', jac=grad, options={'maxiter':2000}, #bounds=self._bounds
                         )
        zeros = np.zeros(self._k_variance_params)
        zeros[0] += np.log(optim.fun)

        return np.random.normal(np.concatenate((optim.x, zeros), axis=0), scale=0.001)

        # return np.random.normal(np.concatenate((inits, zeros), axis=0), scale=0.1)

    def _set_bounds(self):
        self._bounds = []
        self._bounds += [(-5, -2), (-3, -0.5), (-1, 0)]
        self._bounds += [(1, 7), (-3, 0), (-7, -3)]
        self._bounds += [(-3, 1), (-2, 1), (-3, 0)]
        self._bounds += [(-1, 1), (-1, 1), (-2, 0)]
        self._bounds += [(-6, -1), (-3, 0), (-2, 0)]
        self._bounds += [(-4, 4), (-4, 4), (-4, 4), (-4, 4)]


    def _config(self):
        self._k_variance_params = self.dgr_v2 + self.dgr_u2 + 1 if self.mle else 0
        self._k_params = self._k_svi_params + self._k_variance_params
        self._set_bounds()
        self._objf = self._objf_mle if self.mle else self._objf_l2

        if self._inits is None:
            self._inits = self._get_inits_mle() if self.mle else np.random.normal(scale=0.1, size=self._k_params)

        if self.mle:
            self._bounds += [[-30, 5] for i in range(self._k_variance_params)]

    def fit(self, mle=False, inits=None):
        self._inits = inits
        self.mle = mle
        self._config()
        scaling = -1 if self.mle else 1
        func = lambda params: scaling * self._objf(params, False) / self._nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        if self.mle:
            self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad, bounds=self._bounds,
                                           options={'maxiter':2000})
        else:
            self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad, bounds=self._bounds,
                                           options={'maxiter':2000})
        self._rmspe = self._objf(self._optim_results.x, True)



    @staticmethod
    def _curve(tau, beta_min, beta_max, gamma, link_func):
       return link_func(beta_min) + (link_func(beta_max) - link_func(beta_min)) * tau ** np.exp(gamma)

    @staticmethod
    def _min1_plus1_to_R(a):
        return np.log((1 + a) / (1 - a)) / 2

    @staticmethod
    def _R_to_min1_plus1(a):
        return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

    @staticmethod
    def _pos_to_R(a):
        return np.log(a)

    @staticmethod
    def _R_to_pos(a):
        return np.exp(a)

    @staticmethod
    def _R_to_R(a):
        return a


    def _params_to_matrices(self, params, tau, tau_pillars, x_pillars, mle:bool=False):
        a = self._curve(tau, *params[self._idx_a], self._R_to_pos)
        b = self._curve(tau, *params[self._idx_b], self._R_to_pos)
        rho = self._curve(tau, *params[self._idx_rho], self._R_to_min1_plus1)
        m = self._curve(tau, *params[self._idx_m], self._R_to_R)
        s2 = self._curve(tau, *params[self._idx_s2], self._R_to_pos)
        alphas = params[self._idx_alphas]

        if not mle:
            return a, b, rho, m, s2, alphas
        else:
            U = np.diag(self._R_to_pos(self._polynomial(tau_pillars, params[self._idx_u2], self.dgr_u2)))
            tmp = np.concatenate((np.zeros(1), params[self._idx_v2]), axis=0)
            V = np.diag(self._R_to_pos(self._polynomial(x_pillars, tmp, self.dgr_v2)))
            return a, b, rho, m, s2, alphas, U, V

    def _objf_l2(self, transformed_params, return_rmspe:bool=False):
        args = self._params_to_matrices(transformed_params, self.tau, self._tau_pillars, self._x_pillars, False)
        y_hat = self.surface(self.tau, self.x, *args)
        eps = self.vec(self.y - y_hat)[self.loc_notna]
        sse = np.sum(eps**2).flatten()[0]
        msre = ((self.vec(self.y/y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) *100
        return sse if not return_rmspe else rmspe

    def _objf_mle(self, transformed_params, return_rmspe: bool = False):
        a, b, rho, m, s2, alpha, U, V = self._params_to_matrices(transformed_params, self.tau, self._tau_pillars, self._x_pillars, True)
        y_hat = self.surface(self.tau, self.x, a, b, rho, m, s2, alpha)
        eps = self.vec(self.y - y_hat)[self.loc_notna]
        Sigma = self._S @ np.kron(U, V) @ self._S.N
        diagSigma = np.diag(Sigma)
        invSigma = np.diag(1 / diagSigma)
        logdetSigma = np.log(diagSigma).sum()
        LL = - 0.5 * (self._nobs * np.log(2 * np.pi) + logdetSigma + eps.N @ invSigma @ eps)
        msre = ((self.vec(self.y / y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) * 100
        return LL.flatten()[0] if not return_rmspe else rmspe



import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from models.depreciated.curve_models import SviCurve
from joblib import load

#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent.parent.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
HUNDRED = 100
days = 365
set_theme()
min_ttm = 5/days
max_ttm = 1
cm = diverge_map(high=colors[0], low=colors[2])

ticker = 'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH'

df_vols_cube = load(os.path.join(basefolder, 'main_runners', f'{ticker}.pkl'))
moneyness = df_vols_cube.columns[1:].to_numpy().astype(float)




tmp = df_vols_cube.groupby(by=[df_vols_cube['CLOSEST_EXPIRY']]).mean().reset_index()
tmp = df_vols_cube.loc['2021-09-13']
t = tmp.index.unique()[0]

fig, ax = plt.subplots(figsize=(10,10))
for i in range(11):
    ax.plot(moneyness.flatten(), np.array(tmp.iloc[i, 1:]), color=cm(tmp.iloc[i, 0]))#, label=list_T[i])
ax.legend(ncol=3)
plt.tight_layout()
plt.show()


df_params = pd.DataFrame(columns=['Expiry', r'$a$', r'$b$', r'$\rho$', r'$m$', r'$s^2$', 'RMSE'])
df_params['Expiry'] = tmp['CLOSEST_EXPIRY'].copy()
i = 0
fig, ax = plt.subplots(figsize=(10, 10))
for tau in tmp.CLOSEST_EXPIRY.values.flatten():
    y = tmp[tmp.CLOSEST_EXPIRY == tau].values.flatten()[1:].astype(float)
    loc = np.invert(np.isnan(y))
    model = SviCurve(y[loc], moneyness[loc])
    model.fit()
    results = np.concatenate((np.array([tau]), model._params.flatten(), np.array([model._optim_results.fun])))
    df_params[(df_params.Expiry == tau)] = results
    ax.scatter(moneyness[loc].flatten(), y[loc], color=cm(tau))#, label=list_T[i])
    ax.plot(moneyness[loc].flatten(), model.surface(model.x, *model._params.flatten()), linestyle='dashed', color=cm(tau))
    i += 1
plt.tight_layout()
plt.show()


df_params.set_index('Expiry', inplace=True)
fig, ax = plt.subplots(figsize=(20,10), sharex=True, ncols=3, nrows=2)
axs = ax.flatten()
i = 0
for col in df_params.columns:
    df_params[[col]].plot(ax=axs[i])
    i += 1
plt.tight_layout(), plt.show()


mle = False
tmp = tmp.N if tmp.shape[1] == 1 else tmp

n, p = tmp.shape
p -= 1
ones_n, ones_p = np.ones((n,1)), np.ones((1,p))
tau = tmp['CLOSEST_EXPIRY'].values.reshape(n,1)
y = tmp.values[:, 1:].reshape(n,p) #*HUNDRED
model = SeasonalSviPowerSurface(t, tau * ones_p, ones_n * moneyness, tau*y**2)
model.fit(mle=mle, inits=None)
df_params = pd.DataFrame(index=tau.flatten())
curves = model._params_to_matrices(model._optim_results.x, tau, tau.flatten(), moneyness.flatten(), mle=mle)
for i in range(5):
    df_params = pd.concat([df_params, pd.DataFrame(curves[i], index=tau.flatten()),], axis=1)

if mle:
    tmpU = np.diag(curves[6]).reshape(1, len(tau.flatten()))
    tmpV = np.diag(curves[7]).reshape(1, len(moneyness.flatten()))

print(model._rmspe)


surface = model.surface(model.tau, model.x, *curves[:6])
i = 0
fig, ax = plt.subplots(figsize=(10, 10))
for tau in tmp.CLOSEST_EXPIRY.values.flatten():
    ax.scatter(moneyness.flatten(), (tau*y[i]**2), color=cm(tau))#, label=list_T[i])
    ax.plot(moneyness.flatten(), surface[i], linestyle='dashed', color=cm(tau))
    i += 1
plt.tight_layout()
plt.show()


surface = model.surface(model.tau, model.x, *curves[:6])
i = 0
tau = tmp.CLOSEST_EXPIRY.values.flatten()
fig, ax = plt.subplots(figsize=(10, 10))
for m in moneyness:
    ax.scatter(tau, tau*y[:, i]**2, color=cm(np.linspace(0, 1, 11)[i]))#, label=list_T[i])
    ax.plot(tau, surface[:, i], linestyle='dashed', color=cm(np.linspace(0, 1, 11)[i]))
    i += 1
plt.tight_layout()
plt.show()


surface = model.surface_deseasoned(model.tau, model.x, *curves[:6])
i = 0
tau = tmp.CLOSEST_EXPIRY.values.flatten()
fig, ax = plt.subplots(figsize=(10, 10))
for m in moneyness:
    ax.scatter(tau, tau*y[:, i]**2, color=cm(np.linspace(0, 1, 11)[i]))#, label=list_T[i])
    ax.plot(tau, surface[:, i], linestyle='dashed', color=cm(np.linspace(0, 1, 11)[i]))
    i += 1
plt.tight_layout()
plt.show()

i = 0
fig, ax = plt.subplots(figsize=(10, 10))
for tau in tmp.CLOSEST_EXPIRY.values.flatten():
    ax.scatter(moneyness.flatten(), tau*y[i]**2, color=cm(tau))#, label=list_T[i])
    ax.plot(moneyness.flatten(), surface[i], linestyle='dashed', color=cm(tau))
    i += 1
plt.tight_layout()
plt.show()

surface = model.seasons(model.tau, model.x, *curves[:6])
i = 0
tau = tmp.CLOSEST_EXPIRY.values.flatten()
fig, ax = plt.subplots(figsize=(10, 10))
for m in moneyness:
    # ax.scatter(tau, y[:, i], color=cm(np.linspace(0, 1, 11)[i]))#, label=list_T[i])
    ax.plot(tau, surface[:, i], linestyle='dashed', color=cm(np.linspace(0, 1, 11)[i]))
    i += 1
plt.tight_layout()
plt.show()


print(model._optim_results)

df_params.columns = [r'$a$', r'$b$', r'$\rho$', r'$m$', r'$s^2$',]
fig, ax = plt.subplots(figsize=(20,10), sharex=True, ncols=3, nrows=2)
axs = ax.flatten()
i = 0
for col in df_params.columns:
    df_params[[col]].plot(ax=axs[i])
    i += 1
plt.tight_layout(), plt.show()
# #
#
#
# def func(x, rho0, rhom, a):
#     return rho0 + (rhom - rho0)*(np.sign(x)*x)**a
# time = np.linspace(-1, 1, 100)
# fig, ax = plt.subplots(figsize=(10,10))
#
# gamma = 1.5
# ax.plot(time, func(time, 1, 2, gamma))
# plt.tight_layout()
# plt.show()
