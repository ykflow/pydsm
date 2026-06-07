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

M = 1
n_jobs = 1
p_max = 5
N_for = 2500
N_fit = [500]
x_pillars = np.empty((2,1))
tau_pillars = np.empty((2,1))
means = [#'linear',
         'exponential']
fit = True
collapse = True
ekf_and_cekf = True
# q = 0.01/ (np.arange(1, 7)[::-1][:p_max].reshape(p_max,1) *0.025/10 )
q = ['2/3', '4/5', '1', '4/3', '2']
for mean in means:
    for N in N_fit:
        for p in range(1, p_max+1):
            print(p)
            x_grid = np.linspace(-0.4, 0.4, 1, endpoint=True).round(4).reshape(1, 1)  # moneyness-space
            tau_grid = np.linspace(0, 2, p + 1, endpoint=True).round(4)[1:].reshape(p, 1)  # maturity-space
            mce = MonteCarloExperiments(x_grid, tau_grid, N, N_for)
            mce.specify_experiment(fit=fit, mean=mean, variance='diag', missing_rate=0.,
                                   M=M, collapse=collapse, k_factors_linear=p, x_pillars=x_pillars, tau_pillars=tau_pillars)
            mce.perform_expirements(n_jobs=n_jobs, ekf_and_cekf=ekf_and_cekf)
            dump(mce.dict_mc_summary, os.path.join(picklefolder, f'(e)kf_vs_cekf_N={N}_p={p}_mean={mean}.pkl'))

for mean in means:
    fig, ax = plt.subplots(figsize=(20,15), ncols=p_max, nrows=p_max)
    axs = ax.flatten()
    for N in N_fit:
        for p in range(p_max):
            dict_p = load(os.path.join(picklefolder, f'(e)kf_vs_cekf_N={N}_p={p+1}_mean={mean}.pkl'))
            for i in range(p + 1):
                if N == min(N_fit):
                    min_ = np.nanmin(dict_p['CEKF_False_mc_results_summary']['mse_f'][:-1, i]) * 0.99
                    ax[p, i].axline((min_, min_), slope=1, color=colors[5])
                    ax[p, i].axhline(y=min_, color=colors[5])
                    ax[p, i].axvline(x=min_, color=colors[5])
                # ax[p, i].scatter(dict_p['CEKF_False_mc_results_summary']['mse_f'][:, i],
                #                  dict_p['CEKF_True_mc_results_summary']['mse_f'][:, i], alpha=0.5)

                ax[p, i].scatter(dict_p['CEKF_False_mc_results_summary']['mse_f'][:-1, i],
                                 dict_p['CEKF_True_mc_results_summary']['mse_f'][:-1, i], color=colors[i])

                # if i == 0:
                ax[p, i].set_title(f'$f_{{{i + 1}}}$ with $q={{{q[i]}}}$', size=20)
                # else:
                #     ax[p, i].set_title(f'$f_{{{i + 1}}}$ with $q=1/{{{i + 1}}}$', size=20)

                if i == 0:
                        label = f'$p=k={{{p + 1}}}$ \n KF MSFE' if mean == 'linear' else f'$p=k={{{p + 1}}}$ \n EKF MSFE'
                        ax[p, i].set_ylabel(label)

                if p == p_max-1: ax[p, i].set_xlabel('CEKF MSFE')
                ax[p, i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax[p, i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

                for i in range(p + 1, p_max):
                    ax[p, i].set_axis_off()

    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(plotfolder, f'fig3_(e)kf_vs_cekf_{mean}.pdf'))
