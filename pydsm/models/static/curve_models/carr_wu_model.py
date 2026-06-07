import numpy as np
from scipy.optimize import minimize, approx_fprime
from datetime import datetime


class CarrWuCurveModel:
    def __init__(self, y:np.array, x:np.array, tau):
        self.y = y
        self.x = x
        self.tau = tau
        self.n = self.y.shape[0]
        self.k = 4

    @staticmethod
    def curve(x, tau, kappa, omega, v, rho):
        a = -2 * (1 - 2 * kappa * tau - omega * rho * v * tau) / (omega ** 2 * tau ** 2)
        b = - rho * v / omega
        c = ((1 - rho ** 2) * (v ** 2) / (omega ** 2)) + (1 - 2 * kappa * tau - omega * rho * v * tau) ** 2 / (
                    omega ** 4 * tau ** 2)
        I2 = a + 2 * np.sqrt((x - b) ** 2 + c) / tau
        return np.sqrt(I2)

    def fit(self):
        self._config()
        scaling = 100
        func = lambda params: self._objf(params) * scaling
        grad = lambda params: approx_fprime(params, func, 6.5e-6)

        self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad)
        self._params = self._untransform_params(self._optim_results.x)
        self._rmspe = self._objf(self._optim_results.x, True)

    def _config(self):
        self._inits = np.random.normal(scale=0.1, size=self.k)

    @staticmethod
    def _min1_plus1_to_R(a):
        return np.log((1+a)/(1-a))/2

    @staticmethod
    def _R_to_min1_plus1(a):
        return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

    @staticmethod
    def _pos_to_R(a):
        return np.log(a)

    @staticmethod
    def _R_to_pos(a):
        return np.exp(a)

    def _untransform_params(self, trans_params):
        kappa, trans_omega, trans_v, trans_rho = trans_params
        omega = self._R_to_pos(trans_omega)
        v = self._R_to_pos(trans_v)
        rho = self._R_to_min1_plus1(trans_rho)
        return np.array([kappa, omega, v, rho])

    def _objf(self, transformed_params, return_rmspe=False):
        untransformed_params = self._untransform_params(transformed_params)
        y_hat = self.curve(self.x, self.tau, *untransformed_params)
        loss = (self.y - y_hat) ** 2
        rmspe = np.sqrt(((self.y/y_hat - 1)**2).mean()) * 100
        return loss.mean() if not return_rmspe else rmspe




import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from data_utils.build_ts_cubes import build_ts_cube


#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
HUNDRED = 100
days = 365
set_theme()
min_ttm = 5/days
max_ttm = 1
cm = diverge_map(high=colors[2], low=colors[0])
moneyness_grid = np.linspace(-1, 1, 41, endpoint=True).round(2) # 0.05 step size
maturity_grid = np.linspace(0, 1, 25, endpoint=True).round(4) #bi-weekly
maturity_grid = np.linspace(0, 1, 49, endpoint=True).round(4) #weekly
maturity_grid = np.linspace(0, 1, 13, endpoint=True).round(4) #bi-weekly


filtered_dir = 'C:/Users/PI26UT/Documents/Data/ComdtyVols/Filtered Data'
# ticker = 'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH'
ticker = 'Crude Futures_Brent 1st Line_BRENT'
# ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
cols = ['CLOSEST_EXPIRY'] + [-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4]#np.linspace(-.4, .4, 17, endpoint=True).round(2).tolist()

df_vols = pd.read_csv(os.path.join(filtered_dir, f'{ticker}.csv'), sep=';')
df_vols_cube = build_ts_cube(df_vols, maturity_grid, moneyness_grid)[cols]
# df_vols_cube = df_vols_cube[df_vols_cube.CLOSEST_EXPIRY.isin(np.linspace(0, 1, 25, endpoint=True).round(4)[1:])]

if ticker == 'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH':
    loc = df_vols_cube.index <= '20200910'
    df_vols_cube.iloc[loc, 1:] *= 10


moneyness = df_vols_cube.columns[1:].to_numpy().astype(float)
df_params = pd.DataFrame(index=df_vols_cube.index, columns=['Expiry', r'$\kappa$', r'$\omega$', r'$v$', r'$\rho$', 'RMSPE'])
df_params['Expiry'] = df_vols_cube['CLOSEST_EXPIRY'].copy()

for t in df_vols_cube.index.unique():
    tmp = pd.DataFrame(df_vols_cube.loc[t])
    tmp = tmp.T if tmp.shape[1] == 1 else tmp
    for tau in tmp.CLOSEST_EXPIRY.values.flatten():
        y = tmp[tmp.CLOSEST_EXPIRY == tau].values.flatten()[1:].astype(float)
        loc = np.invert(np.isnan(y))
        model = CarrWuCurveModel(y[loc], moneyness[loc], tau)
        model.fit()
        results = np.concatenate((np.array([tau]), model._params.flatten(), np.array([model._rmspe])))
        df_params[(df_params.index ==t) & (df_params.Expiry == tau)] = results
        print(t, tau, model._rmspe)


cm = diverge_map(high=colors[2], low=colors[0])
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,10), sharex=True)
axs = ax.flatten()
i = 0
for param in [ r'$\kappa$', r'$\omega$', r'$v$', r'$\rho$', 'RMSPE']:
    tmp = df_params[['Expiry', param]].reset_index().pivot(index='DATE', columns='Expiry', values=param)
    tmp.rolling(5).mean().plot(ax=axs[i], legend=False, color=cm(tmp.columns), linewidth=1)
    tmp.median(axis=1).plot(ax=axs[i], legend=False, color='black')
    axs[i].set_title(param, size=25)
    axs[i].set_xlim(tmp.index.min(), tmp.index.max())
    axs[i].set_xlabel('')
    i +=1
axs[0].legend(ncols=6, prop={'size': 8}, loc='upper left')
plt.tight_layout()
plt.show()


k = np.linspace(-.4, .4, 100)
kappa, omega, v, rho = -1.17714442,  0.50291809,  0.17093611, -0.99998237

# def carr_wu_curve_basic(k, tau, a, b, c):
#     I2 = a + 2 * np.sqrt((k-b)**2 + c) / tau
#     return np.sqrt(I2)


# def carr_wu_curve_abc(k, tau, kappa, omega, v, rho):
#     a = 0.25*(omega**2 * tau**2)
#     b = 1 - 2 * kappa - omega * rho * v * tau
#     c = v**2 + 2 * omega * rho * v * k + omega**2 * k**2
#
#     tmp = np.sqrt((b**2 - 4*a*c))/(2*a)
#     I2min = (-b - tmp)/(2*a)
#     I2plus = (-b + tmp)/(2*a)
#     return np.sqrt(I2min), np.sqrt(I2plus)


