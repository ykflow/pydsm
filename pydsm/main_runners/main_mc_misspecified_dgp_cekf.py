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
picklefolder_for_fit = os.path.join(picklefolder, 'mc_misspecified_fit=True')
picklefolder_for_no_fit = os.path.join(picklefolder, 'mc_misspecified_fit=False')
set_theme()

ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
N_fit = 500
k_factors = 10
moneyness = np.array([-0.4, -0.3,  -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1,  0.2,  0.3, 0.4])
maturities = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
df_vols_cube = import_ts_ivs(moneyness, maturities, ticker)
df_vols_cube.iloc[:, 1:] **= 2

x_knots = np.array([-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
tau_knots = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12])
cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])

mean, variance = 'carr_wu_disprop_s2', 'mvn-spline'
model_name = f'{mean}_{variance}.pkl'
dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
model = DynamicSurfaceModels(df_surface=df_vols_cube)
model.specify_measurement_equation(mean=mean, variance=variance,
                                   moneyness_pillars=x_knots, maturity_pillars=tau_knots,
                                   cyclical_knots=cyclical_knots)
model._set_model_params(*dict_results['set_params'])

try:
    model.run_filter()
except:
    model.run_filter()


# model.fit(cross_sectional=True)
#
# f_hat, rmspe = pd.DataFrame(model.f_cs), pd.DataFrame(model._rmspe_cs)
# f_hat.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
# # rmspe.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
#
# # Q1 = f_hat.quantile(0.25)
# # Q3 = f_hat.quantile(0.75)
# # IQR = Q3 - Q1
f_hat = pd.DataFrame(model.a_t[:, :, 0])
f_true = f_hat.iloc[:N_fit]#[~((f_hat < (Q1 - 1.5 * IQR)) |(f_hat > (Q3 + 1.5 * IQR))).any(axis=1)].reset_index(drop=True).iloc[:N_fit]
df_stats = pd.concat([f_true.describe(), pd.DataFrame(f_true.diff().var()).T]).iloc[1:, :]

# f_true.plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true.iloc[:, :4].plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true.iloc[:, 4:6].plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true.iloc[:, 6:10].plot(figsize=(20,10)), plt.tight_layout(), plt.show()
f_true = f_true.values.reshape(N_fit, f_true.shape[1], 1)
dump(f_true, os.path.join(picklefolder_for_fit, 'f_true_data.pkl'))
dump(f_true, os.path.join(picklefolder, 'f_true_data.pkl'))


dict_q = dict()
dict_q['low'] = [np.array([[-10, 1],[0, 1], [10, 1]]), np.array([[0, 0.015],[10, 0.015]])]
dict_q['high'] = [np.array([[-10, 1],[0, 1], [10, 1]]), np.array([[0, 0.005],[10, 0.005]])]
x_pillars = np.array([[-0.2, 2], [-0.1, 1.25], [0, 1], [0.1, 1.1], [0.2, 1.4]])
plt.plot(x_pillars[:, 0], x_pillars[:, 1]), plt.show()

tau_pillars = np.array([[1/12, 5], [3/12, 2], [6/12, 1], [9/12, 0.8], [12/12, 0.6], [18/12, 0.7]])
tau_pillars[:, 1] = tau_pillars[:, 1] / 5 * 0.25 /100 * 2
plt.plot(tau_pillars[:, 0], tau_pillars[:, 1]), plt.show()
dict_q['mvn'] = [x_pillars, tau_pillars]

dict_grid_tau = dict()
dict_grid_tau['monthly'] = np.linspace(0, 2, 25, endpoint=True).round(4)[1:].reshape(24,1)
dict_grid_tau['bi-weekly'] = np.linspace(0, 2, 49, endpoint=True).round(4)[1:].reshape(48,1)

dict_grid_x = dict()
dict_grid_x['short'] = np.array([-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2]).reshape(9,1)
dict_grid_x['long'] = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]).reshape(13,1)

M = 100
FIT = True
COLLAPSE = True
N_for = 0
n_jobs = 10
for MISSING in [False,
                True
                ]:
    for mean, k, in zip(['carr_wu_disprop_s2'], [k_factors]):
        for key_q in ['mvn']:
            for key_tau in ['monthly']:
                for key_x in ['long']:
                    x_pillars, tau_pillars = dict_q[key_q]
                    tau_grid = dict_grid_tau[key_tau]
                    x_grid = dict_grid_x[key_x]
                    p = len(x_grid) * len(tau_grid)
                    label1 = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}_missing={MISSING}_combined.pkl'
                    label2 = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}_missing={MISSING}_summary.pkl'
                    label3 = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}_missing={MISSING}_all.pkl'
                    print(label1)
                    mce = MonteCarloExperiments(x_grid, tau_grid, N_fit, N_for, f_true[:, :k])
                    mce.specify_experiment(fit=FIT, mean=mean, variance='mvn-spline',
                                           M=M,#+n_jobs,
                                           collapse=COLLAPSE,
                                           missing_moneyness=MISSING,
                                           missing_rate=0.,
                                           x_pillars=x_pillars, tau_pillars=tau_pillars)
                    mce.perform_expirements(n_jobs=n_jobs, ekf_and_cekf=False)
                    # dump(mce._mc_results_combined, f'{label1}.pkl')
                    # dump(mce._mc_results_summary, f'{label2}.pkl')
                    # dump(mce._mc_results, f'{label3}.pkl')
                    dump(mce._mc_results_combined, os.path.join(picklefolder, label1))
                    dump(mce._mc_results_summary, os.path.join(picklefolder, label2))
                    dump(mce._mc_results, os.path.join(picklefolder, label3))



# M = 50
# FIT = False
# COLLAPSE = True
# N_for = 0
# n_jobs = 5
# for mean, k, in zip(['carr_wu_standard', 'carr_wu_prop2_s2'], [4, 11]):
#     for key_q in dict_q.keys():
#         for key_tau in dict_grid_tau.keys():
#             for key_x in dict_grid_x.keys():
#                 x_pillars, tau_pillars = dict_q[key_q]
#                 tau_grid = dict_grid_tau[key_tau]
#                 x_grid = dict_grid_x[key_x]
#                 p = len(x_grid) * len(tau_grid)
#                 label = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}.pkl'
#                 print(label)
#                 mce = MonteCarloExperiments(x_grid, tau_grid, N_fit, N_for, f_true[:, :k])
#                 mce.specify_experiment(fit=FIT, mean=mean, variance='mvn-spline',
#                                        M=M+n_jobs, collapse=COLLAPSE,
#                                        missing_moneyness=False,
#                                        missing_rate=0.,
#                                        x_pillars=x_pillars, tau_pillars=tau_pillars)
#                 mce.perform_expirements(n_jobs=n_jobs, ekf_and_cekf=False)
#                 dump(mce._mc_results_summary, os.path.join(picklefolder, label))

# FIT = False
# results = []
# for mean, k, in zip(['carr_wu_standard', 'carr_wu_prop2_s2'], [4, 11]):
#     df_results = pd.DataFrame()
#     for key_tau in dict_grid_tau.keys():
#         for key_x in dict_grid_x.keys():
#             df_tmp = pd.DataFrame()
#             for key_q in dict_q.keys():
#                 tau_grid = dict_grid_tau[key_tau]
#                 x_grid = dict_grid_x[key_x]
#                 x_pillars, tau_pillars = dict_q[key_q]
#                 p = len(x_grid) * len(tau_grid)
#                 label = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}.pkl'
#                 print(label)
#                 df = pd.DataFrame(load(os.path.join(picklefolder, label)))
#
#                 # print(df.T)
#                 # print('\n')
#
#                 df_tmp = pd.concat([df_tmp, df.T], axis=1)
#             df_results = pd.concat([df_results, df_tmp], axis=0)
#     results.append(df_results)


# results = []
# for mean, k, in zip(['carr_wu_prop2_s2'], [11]):
#     df_results = pd.DataFrame()
#     for key_tau in ['monthly']:
#         for key_x in ['long']:
#             for key_q in ['mvn']:
#                 df_tmp = pd.DataFrame()
#                 x_pillars, tau_pillars = dict_q[key_q]
#                 tau_grid = dict_grid_tau[key_tau]
#                 x_grid = dict_grid_x[key_x]
#                 p = len(x_grid) * len(tau_grid)
#                 label1 = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}_missing={MISSING}_combined.pkl'
#                 label2 = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}_missing={MISSING}_summary.pkl'
#                 print(label1)
#                 df = pd.DataFrame(load(os.path.join(picklefolder, label2)))
#
#             # print(df.T)
#             # print('\n')
#
#                 df_tmp = pd.concat([df_tmp, df.T], axis=1)
#             df_results = pd.concat([df_results, df_tmp], axis=0)
#     results.append(df_results)


# fig, ax = plt.subplots(figsize=(20,15), ncols=p_max, nrows=p_max, sharey=True, sharex='col')
# axs = ax.flatten()
# for N in N_fit:
#     for p in range(p_max):
#         dict_p = load(os.path.join(picklefolder, f'kf_vs_cekf_N={N}_p={p+1}.pkl'))
#         for i in range(p + 1):
#             # if N == min(N_fit):
#             #     min_ = dict_p['CEKF_False_mc_results_summary']['mse_f'][:, i].min() * 0.99
#             #     ax[p, i].axline((min_, min_), slope=1, color=colors[5])
#             #     ax[p, i].axhline(y=min_, color=colors[5])
#             #     ax[p, i].axvline(x=min_, color=colors[5])
#             # ax[p, i].scatter(dict_p['CEKF_False_mc_results_summary']['mse_f'][:, i],
#             #                  dict_p['CEKF_True_mc_results_summary']['mse_f'][:, i], alpha=0.5)
#
#             ax[p, i].hist(#dict_p['CEKF_False_mc_results_summary']['mse_f'][:, i],
#                              dict_p['CEKF_True_mc_results_summary']['mse_f'][:, i], alpha=0.5)
#
#             if i == 0:
#                 ax[p, i].set_title(f'$f_{{{i + 1}}}$ with $q={{{i + 1}}}$', size=20)
#             else:
#                 ax[p, i].set_title(f'$f_{{{i + 1}}}$ with $q=1/{{{i + 1}}}$', size=20)
#
#             if i == 0: ax[p, i].set_ylabel(f'$p=k={{{p + 1}}}$ \n KF MSFE')
#             if p == p_max-1: ax[p, i].set_xlabel('CEKF MSFE')
#             # ax[p, i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#             ax[p, i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#
#             for i in range(p + 1, p_max):
#                 ax[p, i].set_axis_off()
#
# plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join(plotfolder, 'fig3_kf_vs_cekf.pdf'))
