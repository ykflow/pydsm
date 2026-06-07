import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from plotting_tools.set_plotting_theme import set_theme, colors
from monte_carlo.monte_carlo_expirements import MonteCarloExperiments
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from config_utils.config_utils_main import import_ts_ivs
from joblib import dump, load

#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
picklefolder = os.path.join(basefolder, 'pickle_folder', )
set_theme()

ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
N_fit = 500
moneyness = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
maturities = np.linspace(0, 49/24, 50, endpoint=True).round(4)[2:]

df_vols_cube = import_ts_ivs(moneyness, maturities, ticker)
df_vols_cube.iloc[:, 1:] **= 2
model = DynamicSurfaceModels(df_surface=df_vols_cube)
model.specify_measurement_equation(mean='model_for_mc')
model.fit(cross_sectional=True)
f_hat, rmspe = pd.DataFrame(model.f_cs), pd.DataFrame(model._rmspe_cs)
f_hat.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
rmspe.plot(figsize=(20,10)), plt.tight_layout(), plt.show()

Q1 = f_hat.quantile(0.25)
Q3 = f_hat.quantile(0.75)
IQR = Q3 - Q1
f_true = f_hat[~((f_hat < (Q1 - 1.5 * IQR)) |(f_hat > (Q3 + 1.5 * IQR))).any(axis=1)].reset_index(drop=True).iloc[:N_fit]

f_true.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true.iloc[:, 4:7].plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true.iloc[:, :4].plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true.iloc[:, 7:].plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true = f_true.values.reshape(N_fit, f_true.shape[1], 1)


dict_q = dict()
dict_q['high'] = [np.array([[-10, 1],[0, 1], [10, 1]]), np.array([[0, 0.005],[10, 0.005]])]
dict_grid_tau = dict()
dict_grid_tau['cw'] = np.array([1/12, 3/12, 6/12, 1, 2, 3, 4, 5]).reshape(8,1)
dict_grid_x = dict()
dict_grid_x['cw'] = np.array([-0.2, -0.1, 0, 0.1, 0.2]).reshape(5,1)

M = 100
FIT = False
COLLAPSE = False
UNSCENTED = False
N_for = 0
n_jobs = 10
CEKF_AND_UKF = True
for mean, k, in zip(['carr_wu_standard'], [4]):
    for key_q in dict_q.keys():
        for key_tau in dict_grid_tau.keys():
            for key_x in dict_grid_x.keys():
                x_pillars, tau_pillars = dict_q[key_q]
                tau_grid = dict_grid_tau[key_tau]
                x_grid = dict_grid_x[key_x]
                p = len(x_grid) * len(tau_grid)
                label = f'real_dgp_cekf_vs_ukf_mean={mean}_p={p}_fit={FIT}.pkl'
                print(label)
                mce = MonteCarloExperiments(x_grid, tau_grid, N_fit, N_for, f_true[:, :k])
                mce.specify_experiment(fit=FIT, mean=mean, variance='mvn-spline',
                                       missing_rate=0, missing_moneyness=False,
                                       M=M+n_jobs, collapse=COLLAPSE, unscented=UNSCENTED,
                                       x_pillars=x_pillars, tau_pillars=tau_pillars)
                mce.perform_expirements(n_jobs=n_jobs, ekf_and_cekf=False, ukf_and_cekf=CEKF_AND_UKF)
                dump(mce.dict_mc_summary, os.path.join(picklefolder, label))

df_results = pd.DataFrame()
for mean, k, in zip(['carr_wu_standard'], [4]):
    for key_q in dict_q.keys():
        for key_tau in dict_grid_tau.keys():
            for key_x in dict_grid_x.keys():
                x_pillars, tau_pillars = dict_q[key_q]
                tau_grid = dict_grid_tau[key_tau]
                x_grid = dict_grid_x[key_x]
                p = len(x_grid) * len(tau_grid)
                label = f'real_dgp_cekf_vs_ukf_mean={mean}_p={p}_fit={FIT}.pkl'
                print(label)
                tmp =load(os.path.join(picklefolder, label))
                for algo in ['CEKF', 'UKF']:
                    print(algo)
                    df = pd.DataFrame(tmp[f'{algo}_mc_results_summary'])
                    df_results = pd.concat([df_results, df.T], axis=1)

