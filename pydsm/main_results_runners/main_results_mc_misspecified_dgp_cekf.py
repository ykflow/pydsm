import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_tools.set_plotting_theme import set_theme, colors
from joblib import dump, load
from scipy.interpolate import interp1d


#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
picklefolder = os.path.join(basefolder, 'pickle_folder', )
mcefolder = os.path.join(picklefolder, 'monte_carlo_evidence')
tablefolder = os.path.join(basefolder, 'table_folder')
set_theme()

M = 100
x_pillars = np.array([[-0.2, 2], [-0.1, 1.25], [0, 1], [0.1, 1.1], [0.2, 1.4]])
tau_pillars = np.array([[1/12, 5], [3/12, 2], [6/12, 1], [9/12, 0.8], [12/12, 0.6], [18/12, 0.7]])
tau_pillars[:, 1] = tau_pillars[:, 1] / 5 * 0.25 /100 * 2

dict_q = dict()
dict_q['low'] = [np.array([[-10, 1],[0, 1], [10, 1]]), np.array([[0, 0.015],[10, 0.015]])]
dict_q['high'] = [np.array([[-10, 1],[0, 1], [10, 1]]), np.array([[0, 0.005],[10, 0.005]])]
dict_q['mvn'] = [x_pillars, tau_pillars]

dict_grid_tau = dict()
dict_grid_tau['monthly'] = np.linspace(0, 2, 25, endpoint=True).round(4)[1:].reshape(24,1)
dict_grid_tau['bi-weekly'] = np.linspace(0, 2, 49, endpoint=True).round(4)[1:].reshape(48,1)

dict_grid_x = dict()
dict_grid_x['short'] = np.array([-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2]).reshape(9,1)
dict_grid_x['long'] = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]).reshape(13,1)



results = []
FIT = True
accuracy_keys = ['mse_f', 'mae_f', 'me_f', 'cvg95%_f']
mean = 'carr_wu_disprop_s2'
key_tau = 'monthly'
key_x = 'long'
key_q = 'mvn'
df_accuracy = pd.DataFrame()
df_mle = pd.DataFrame()
f_true = load(os.path.join(mcefolder, 'f_true_data.pkl'))
fig, ax = plt.subplots(figsize=(20*1.25,10*1.25), ncols=4, nrows=3)
axs = ax.flatten()
titles = [r'$\kappa_{1,t}$', r'$\omega^{\star}_{1,t}$', r'$\nu^{\star}_{t}$', r'$\rho^{\star}_{t}$',
          r'${\eta}_{1,t}$', r'${\eta}_{2,t}$',
          r'$\delta_{1,t}$', r'$\delta^\ast_{1,t}$', r'$\delta_{2,t}$', r'$\delta^\ast_{2,t}$',
          r'$v^2(\tau)$ $(\times 1000)$', r'$u^2(x)$']
xlabels = [r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$',
          r'$\tau$', r'$x$']

M = 0
# order_f = [0, 1, 2, 3, 4, 5, 7, 8, 9, 6]
for MISSING in [False, True]:
    df_accuracy_tmp = pd.DataFrame()
    x_pillars, tau_pillars = dict_q[key_q]
    tau_grid = dict_grid_tau[key_tau]
    x_grid = dict_grid_x[key_x]
    p = len(x_grid) * len(tau_grid)
    label = f'real_dgp_cekf_mean={mean}_q={key_q}_tau={key_tau}_x={key_x}_p={p}_fit={FIT}_missing={MISSING}_combined.pkl'
    print(label)
    dict_tmp = load(os.path.join(mcefolder, label))

    for key in accuracy_keys:
        df_accuracy_tmp = pd.concat([df_accuracy_tmp, pd.DataFrame(dict_tmp[key][M:].mean(axis=0)).T], axis=0)
    df_accuracy = pd.concat([df_accuracy, df_accuracy_tmp], axis=0)
    tmp_mle = dict_tmp['mle'][M:, 10:]
    tmp_mle[:, 4:] *= 1000
    df_mle = pd.concat([df_mle, pd.DataFrame(tmp_mle).describe(percentiles=[.10, .5, .9])], axis=0)

    f_hat = dict_tmp['f_hat'][M:, :, :]
    avg = np.mean(f_hat, axis=0)
    lb = np.quantile(f_hat, 0.025, axis=0)
    ub = np.quantile(f_hat, 0.975, axis=0)
    if MISSING:

        for i in range(f_true.shape[1]):
            axs[i].fill_between(np.arange(500), lb[:, i], ub[:, i], alpha=0.5, color=colors[2], label='MC CI(95%)')
            axs[i].plot(f_true[:, i], color='black', linewidth=1, linestyle='dashed', label=r'True',)
            axs[i].plot(avg[:, i], linewidth=1, color=colors[2], label=r'MC Avg.')
            axs[i].set_xlim(0, 500)
            print((f_true[:, i] - avg[:, i]).mean())

        sigma2 = dict_tmp['mle'][M:, f_true.shape[1]:]
        sigma2_x = sigma2[:, :4]
        sigma2_tau = sigma2[:, 4:] *1000
        axs[-2].scatter(tau_pillars[:, 0], tau_pillars[:, 1]*1000, color='black', s=10)
        axs[-2].scatter(tau_pillars[:, 0],  sigma2_tau.mean(axis=0) , color=colors[2], s=15, marker='*')
        for i in range(6):
            lb = np.quantile(sigma2_tau, 0.025, axis=0)
            ub = np.quantile(sigma2_tau, 1-0.025, axis=0)
            axs[-2].plot(tau_pillars[i, 0] * np.ones(2), np.array([lb[i], ub[i]]), color=colors[2], linewidth=10, alpha=0.5)

        axs[-1].scatter(x_pillars[:, 0], x_pillars[:, 1], color='black', s=10)
        axs[-1].scatter(np.array([-0.2, -0.1, 0.1, 0.2]),  sigma2_x.mean(axis=0), color=colors[2], s=15, marker='*')
        for i in range(4):
            lb = np.quantile(sigma2_x, 0.025, axis=0)
            ub = np.quantile(sigma2_x, 1-0.025, axis=0)
            axs[-1].plot(np.array([-0.2, -0.1, 0.1, 0.2])[i]* np.ones(2), np.array([lb[i], ub[i]]), color=colors[2], linewidth=10, alpha=0.5)

for i in range(12):
    axs[i].axhline(y=0, color=colors[5], label='_nolegend_')
    axs[i].set_xlabel(xlabels[i])
    axs[i].set_title(titles[i], size=25)

df_accuracy.style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, 'mc_fit_accuracy.tex'))
df_mle.style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, 'mc_fit_mle.tex'))

handles, labels = axs[0].get_legend_handles_labels()
handles = [handles[1], handles[2], handles[0]]
labels = [labels[1], labels[2], labels[0]]
axs[0].legend(handles, labels, ncol=1, prop={'size': 15}, loc='lower left')
plt.tight_layout(w_pad=0.01)
plt.savefig(os.path.join(plotfolder, 'mc_fit_with_missing.pdf'))
# plt.show()

f_true = pd.DataFrame(f_true[:, :, 0])
df_stats = f_true.describe(percentiles=[.10, .5, .9])
df_stats.style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, 'mc_f_true_stats.tex'))