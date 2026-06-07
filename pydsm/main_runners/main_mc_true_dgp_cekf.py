import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from plotting_tools.set_plotting_theme import set_theme, colors
from monte_carlo.monte_carlo_expirements import MonteCarloExperiments
from joblib import dump, load

#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
picklefolder = os.path.join(basefolder, 'pickle_folder', )
set_theme()

M = 10
N_for = 500
N_fit = 500

x_grids = [
    # np.array([-0.2, -0.1, 0., 0.1, 0.2]).reshape(5,1),
           np.array([-0.6, -0.4, -0.2, -0.1, -0.05, -0.025, 0., 0.025, 0.05,  0.1, 0.2, .4, 0.6]).reshape(13,1)
           ]
tau_grid = np.linspace(0, 2, 25, endpoint=True).round(4)[1:].reshape(24, 1)
# tau_grid = np.linspace(0, 1, 13, endpoint=True).round(4)[1:].reshape(12, 1)

x_pillars = np.array([[-0.2, 2], [-0.1, 1.25], [0, 1], [0.1, 1.1], [0.2, 1.4]])
# plt.plot(x_pillars[:, 0], x_pillars[:, 1]), plt.show()

# tau_pillars = np.array([[1/12, 5], [3/12, 2], [6/12, 1], [9/12, 0.8], [12/12, 0.6], [18/12, 0.4],  [24/12, 0.3]])
tau_pillars = np.array([[1/12, 5], [3/12, 2], [6/12, 1], [9/12, 0.8], [12/12, 0.6]])
tau_pillars[:, 1] = tau_pillars[:, 1] / 5 * 0.25 /100
# plt.plot(tau_pillars[:, 0], tau_pillars[:, 1]), plt.show()


cyclical_knots = np.array([1/12, 3/12, 6/12, 9/12])
n_jobs = 1
FIT = False
COLLAPSE = True
for x_grid in x_grids:
    mce = MonteCarloExperiments(x_grid, tau_grid, N_fit, N_for)
    mce.specify_experiment(fit=FIT, mean='carr_wu_prop2', variance='iid', missing_rate=0.,
                           M=M, collapse=COLLAPSE, x_pillars=x_pillars, tau_pillars=tau_pillars, cyclical_knots=cyclical_knots)
    mce.perform_expirements(n_jobs=n_jobs, ekf_and_cekf=False)
    # dump(mce.dict_mc_summary, os.path.join(picklefolder, f'true_dgp_cekf_N={N}_p={p}.pkl'))



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
