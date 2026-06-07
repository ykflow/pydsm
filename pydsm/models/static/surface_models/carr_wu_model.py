import numpy as np
from scipy.optimize import minimize, approx_fprime
from datetime import datetime


class CarrWuSurfaceModel:
    def __init__(self, t, y:np.array, x:np.array, tau):
        self.t = t
        self.y = y
        self.x = x
        self.tau = tau
        self.n = self.y.shape[0]
        self.k = 15
        self._t = self.day_of_year(self.t)

    @staticmethod
    def day_of_year(t: datetime):
        return ((t - datetime(t.year, 1, 1)).days + 1) / 365

    # @staticmethod
    def surface(self, x, tau, kappa, omega, v, rho, eta1, eta2, eta3, nu1, nu2, nu3, nu4, nu5, nu6, nu7, nu8):
        T = self._t + tau
        S = (nu1*np.sin(2*np.pi*T) + nu2*np.cos(2*np.pi*T)
             + nu3*np.sin(4*np.pi*T) + nu4*np.cos(4*np.pi*T)
             + nu5 * np.sin(8 * np.pi * T) + nu6 * np.cos(8 * np.pi * T)
             + nu7 * np.sin(16 * np.pi * T) + nu8 * np.cos(16 * np.pi * T)
             )

        # omega += S

        kappa = kappa*np.exp(-eta1*tau) + S * np.exp(-eta2*tau)
        omega = omega * np.exp(-eta1*tau)
        rho = rho * np.exp(-eta3*tau)
        v = v * np.exp(-eta3 * tau)


        a = -2 * (1 - 2 * kappa * tau - omega * rho * v * tau) / (omega ** 2 * tau ** 2)
        b = - rho * v / omega
        c = ((1 - rho ** 2) * (v ** 2) / (omega ** 2)) + (1 - 2 * kappa * tau - omega * rho * v * tau) ** 2 / (
                    omega ** 4 * tau ** 2)
        I2 = a + 2 * np.sqrt((x - b) ** 2 + c) / tau
        return np.sqrt(I2)

    def fit(self):
        self._config()
        scaling = 100*100
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
        kappa, trans_omega, trans_v, trans_rho, trans_eta1, trans_eta2, trans_eta3, nu1, nu2, nu3, nu4, nu5, nu6, nu7, nu8= trans_params
        omega = self._R_to_pos(trans_omega)
        v = self._R_to_pos(trans_v)
        rho = self._R_to_min1_plus1(trans_rho)
        eta1 = self._R_to_pos(trans_eta1)
        eta2 = self._R_to_pos(trans_eta2)
        eta3 = self._R_to_pos(trans_eta3)

        return np.array([kappa, omega, v, rho, eta1, eta2, eta3, nu1, nu2, nu3, nu4, nu5, nu6, nu7, nu8])

    def _objf(self, transformed_params, return_rmspe=False):
        untransformed_params = self._untransform_params(transformed_params)
        y_hat = self.surface(self.x, self.tau, *untransformed_params)
        loss = (self.y - y_hat) ** 2
        rmspe = np.sqrt(np.nanmean((self.y/y_hat - 1)**2)) * 100
        return np.nanmean(loss) if not return_rmspe else rmspe




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
moneyness_grid = np.linspace(-.5, .5, 41, endpoint=True).round(4) # 0.025 step size
# maturity_grid = np.linspace(0, 25/24, 26, endpoint=True).round(4) #bi-weekly
# maturity_grid = np.linspace(0, 53/52, 54, endpoint=True).round(4)
# maturity_grid = np.linspace(0, 13/12, 14, endpoint=True).round(4) #bi-weekly

maturity_grid = np.linspace(0, 2, 50, endpoint=True).round(4)
moneyness = [-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2]


filtered_dir = 'C:/Users/PI26UT/Documents/Data/ComdtyVols/Filtered Data'
# ticker = 'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH'
# ticker = 'Crude Futures_Brent 1st Line_BRENT'
ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
cols = ['CLOSEST_MONEYNESS'] + maturity_grid.tolist()[2:] #np.linspace(-.4, .4, 17, endpoint=True).round(2).tolist()


df_vols = pd.read_csv(os.path.join(filtered_dir, f'{ticker}.csv'), sep=';')
df_vols_cube = build_ts_cube(df_vols, maturity_grid, moneyness_grid)[cols]
df_vols_cube = df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS.isin(moneyness)]

if ticker == 'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH':
    loc = df_vols_cube.index <= '20200910'
    df_vols_cube.iloc[loc, 1:] *= 10

df_params = pd.DataFrame(index=df_vols_cube.index.unique(), columns=[r'$\kappa$', r'$\omega$', r'$v$', r'$\rho$',
                                                                     r'$\eta_1$', r'$\eta_2$', r'$\nu_1$', r'$\nu_2$',
                                                                     r'$\nu_3$', r'$\nu_4$', r'$\nu_5$', r'$\nu_6$',
                                                                     r'$\nu_7$', r'$\nu_8$','RMSPE'])

# for t in df_vols_cube.index.unique():
#     tmp = pd.DataFrame(df_vols_cube.loc[t])
#     tmp = tmp.T if tmp.shape[1] == 1 else tmp
#     n, p = tmp.shape
#     p -= 1
#     ones_n, ones_p = np.ones((n,1)), np.ones((1,p))
#     tau = tmp.columns[1:].values.reshape(1,p).astype(float)
#     moneyness = tmp.CLOSEST_MONEYNESS.values.reshape(n,1).astype(float)
#     y = tmp.values[:, 1:].reshape(n,p)
#     model = CarrWuSurfaceModel(t, y, ones_p * moneyness, tau * ones_n)
#     model.fit()
#     params = model._params
#     rmspe = model._rmspe
#     df_params.loc[t] = np.concatenate((params, np.array([rmspe])))
#     print(t, rmspe)
#
# cm = diverge_map(high=colors[2], low=colors[0])
# fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20*1.5,10*1.5), sharex=True)
# axs = ax.flatten()
# i = 0
# for param in df_params.columns:
#     tmp = df_params[[param]].copy()
#     tmp.clip(tmp.quantile(0.1), tmp.quantile(0.9), axis=1).rolling(10).mean().plot(ax=axs[i], legend=False, linewidth=1)
#     axs[i].set_title(param, size=25)
#     axs[i].set_xlim(tmp.index.min(), tmp.index.max())
#     axs[i].set_xlabel('')
#     i +=1
# axs[0].legend(ncols=6, prop={'size': 8}, loc='upper left')
# plt.tight_layout()
# plt.show()


tmp = df_vols_cube.groupby(by=[df_vols_cube['CLOSEST_MONEYNESS']]).mean().reset_index()
# tmp = df_vols_cube.loc['2021-09-13']
tmp = df_vols_cube.loc['2021-09-27']
# tmp = df_vols_cube.loc[ '2022-08-11']
t = tmp.index.unique()[0]

n, p = tmp.N.shape
n -= 1
ones_n, ones_p = np.ones((n,1)), np.ones((1,p))
maturities = tmp.N.index[1:].values.reshape(n, 1).astype(float)
y = tmp.N.values[1:, ].reshape(n, p).astype(float)
model = CarrWuSurfaceModel(t, y, ones_n * moneyness, maturities * ones_p)
model.fit()
params = model._params
rmspe = model._rmspe
# df_params.loc[t] = np.concatenate((params, np.array([rmspe])))
print(t, rmspe)


fig, ax = plt.subplots(figsize=(20,10), ncols=2)
surface = model.surface(model.x, model.tau, *model._params)
i = 0
for m in moneyness:
    ax[0].scatter(maturities, y[:, i], color=cm(np.linspace(0, 1, p)[i]))#, label=list_T[i])
    ax[0].plot(maturities, surface[:, i], linestyle='dashed', color=cm(np.linspace(0, 1, p)[i]), label=f'x={m}')
    i += 1


params = model._params.copy()
params[6:] = 0
surface = model.surface(model.x, model.tau, *params)
i = 0
for tau in maturities:
    ax[1].plot(moneyness, surface[i,:], linestyle='dashed', color=cm(tau), label=f'tau={tau}')
    i += 1

ax[0].set_xlabel('Maturity (Years)')
ax[1].set_xlabel('Log-Moneyness')
ax[0].legend(ncol=3, prop={'size': 15})
ax[1].legend(ncol=3, prop={'size': 15})
plt.tight_layout()
plt.show()