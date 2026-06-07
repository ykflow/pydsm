import os

import matplotlib.cm
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from config_utils.config_utils_main import import_ts_ivs
from numba.typed import List
from joblib import dump, load
from measurement_equations.carr_wu_seasonal import seasonal_effect
from measurement_equations.carr_wu_seasonal import map_moments
from calculator.forward_curve_calculator import ForwardCurveCalculator
from calculator.yield_curve_calculator import ZeroRatesCalculator
from pricers.variance_swap_rate import VarianceSwapPricer
from scipy.stats import norm
import arch


def cw_atm(f, tau, add_S=True):
    kappa1, omega1_star, nu_star, rho_star, eta1_tilde, eta2_tilde = f.flatten()[:6]
    omega1, nu, rho, gamma = map_moments(omega1_star, nu_star, rho_star)
    deltas = f[6:]
    exp_eta1_tau = np.exp(eta1_tilde*tau)
    exp_eta2_tau = np.exp(eta2_tilde*tau)
    S = seasonal_effect(deltas, tau)
    kappa = kappa1 * exp_eta1_tau

    if add_S:
        kappa += S * exp_eta2_tau

    return (nu**2) / (1 - 2 * kappa * tau)


#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
picklefolder = os.path.join(basefolder, 'pickle_folder', )
tablefolder = os.path.join(basefolder, 'table_folder', )
HUNDRED = 100
days = 365
set_theme()
cm = diverge_map(high=colors[2], low=colors[0])
start = '20200801'
end = '20240831'
train_end = '20230731'
key_ttm = [0.1667, 0.25, 0.5, 1, 1.5, 2]
dict_key_ttm_labels = dict({0.1667: '2W', 0.25: '3M', 0.5: '6M', 1: '1Y', 1.5: '18M', 2: '2Y'})
dict_key_ttm_clrs = dict({0.1667: 0.1667/2.6, 0.25: 0.5/2.6, 0.5: 1/2.6, 1: 1.5/2.6, 1.5: 2/2.6, 2: 2.5/2.6})
key_ttm_labels = ['2W', '3M', '6M', '1Y', '18M', '2Y']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


cols1 = ['Date', '2024', '2023', 'Avg', 'Low', 'High']
cols2 = ['Date', '2024', 'Avg', 'Low', 'High']
df_us_ng_storage = pd.read_csv(os.path.join(datafolder, 'eia', 'Lower_48_weekly_working_gas_in_underground_storage.csv'), skiprows=4)
df_us_ng_storage_net_change = pd.read_csv(os.path.join(datafolder, 'eia', 'Lower_48_weekly_net_change_in_working_gas_in_underground_storage.csv'), skiprows=4)
df_us_ng_storage.columns = cols1
df_us_ng_storage_net_change.columns = cols2
df_us_ng_storage_net_change['Date'] = pd.to_datetime(df_us_ng_storage_net_change['Date'])
df_us_ng_storage['Date'] = pd.to_datetime(df_us_ng_storage['Date'])
df_us_ng_storage.set_index('Date', inplace=True)
df_us_ng_storage_net_change.set_index('Date', inplace=True)
df_us_ng_storage_net_change.sort_index(inplace=True)
df_us_ng_storage.sort_index(inplace=True)


### CONFIG
ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
x_grid = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
x_knots = np.array([-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
# tau_grid = np.linspace(0, 2, 49, endpoint=True).round(4)[1:]
tau_grid = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
tau_knots = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12])
cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])

## DATA IMPORTS
df_vols_cube = import_ts_ivs(x_grid, tau_grid, ticker)
df_vols_cube = df_vols_cube[(df_vols_cube.index >= start) & (df_vols_cube.index <= end)]
df_vols_cube.iloc[:, 1:] **= 2
timeline = df_vols_cube.index.unique()

df_hhng_spot = pd.read_csv(os.path.join(datafolder, 'spots', 'DHHNGSP.csv'))
df_hhng_spot = df_hhng_spot.set_index(pd.to_datetime(df_hhng_spot.DATE), drop=True).dropna()[['DHHNGSP']]

fcc = ForwardCurveCalculator(os.path.join(datafolder, 'comdty_forwards'))
zrc = ZeroRatesCalculator(os.path.join(datafolder, 'rates'))

df_forwards = fcc.interpolate(tau_grid, timeline)
df_rates = zrc.interpolate(tau_grid, timeline)


cols1 = ['Date', '2024', '2023', 'Avg', 'Low', 'High']
cols2 = ['Date', '2024', 'Avg', 'Low', 'High']
df_us_ng_storage = pd.read_csv(os.path.join(datafolder, 'eia', 'Lower_48_weekly_working_gas_in_underground_storage.csv'), skiprows=4)
df_us_ng_storage_net_change = pd.read_csv(os.path.join(datafolder, 'eia', 'Lower_48_weekly_net_change_in_working_gas_in_underground_storage.csv'), skiprows=4)
df_us_ng_storage.columns = cols1
df_us_ng_storage_net_change.columns = cols2
df_us_ng_storage_net_change['Date'] = pd.to_datetime(df_us_ng_storage_net_change['Date'])
df_us_ng_storage['Date'] = pd.to_datetime(df_us_ng_storage['Date'])
df_us_ng_storage.set_index('Date', inplace=True)
df_us_ng_storage_net_change.set_index('Date', inplace=True)
df_us_ng_storage_net_change.sort_index(inplace=True)
df_us_ng_storage.sort_index(inplace=True)


fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20 * 1.5, 10 * 1.5), sharey=True, height_ratios=[3, 3, 5])
axs = ax.flatten()
i = 0
labels_tmp = ['DITM', 'DITM', 'ITM', 'ATM', 'OTM', 'DOTM', 'DOTM']
for moneyess, atm in zip([-0.4, -0.3, -0.2, -0.1, 0, .1, 0.2, .3, .4],
                         ['DITM', 'DITM', 'ITM', 'ITM', 'ATM', 'OTM', 'OTM', 'DOTM', 'DOTM']):
    tmp = df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS == moneyess].iloc[:, 1:] **.5 * HUNDRED
    for t in tmp.index:
        y = tmp[tmp.index == t]
        nobs = y.shape[1]
        plot = axs[i].scatter([t] * nobs, y.values.flatten(), c=y.columns, s=12, cmap=cm, alpha=0.35, vmin=0, vmax=2.1)
        axs[i].set_title(f'log(K/F)={moneyess} ({atm})', size=30)
        axs[i].set_xlim(df_vols_cube.index.min(), df_vols_cube.index.max())
    print(i)
    i += 1

fig.autofmt_xdate(rotation=45)
cbar1 = fig.colorbar(plot, ax=axs[6], location='bottom', pad=0.225, label='Time-to-Maturity (Years)')
cbar2 = fig.colorbar(plot, ax=axs[7], location='bottom', pad=0.225, label='Time-to-Maturity (Years)')
cbar3 = fig.colorbar(plot, ax=axs[8], location='bottom', pad=0.225, label='Time-to-Maturity (Years)')
cbar1.solids.set(alpha=.8)
cbar2.solids.set(alpha=.8)
cbar3.solids.set(alpha=.8)
for i in [0, 3, 6]:
    axs[i].set_ylabel(r'Implied Vol. $I(x, \tau)$ (%)')
# plt.suptitle(f'{ticker}', size=35)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plotfolder, f'HH_vols.png'))

#
#
fig, (ax1, ax2) = plt.subplots(figsize=(20,10), nrows=2, sharex=True, height_ratios=[1, 3])
fig.subplots_adjust(hspace=0.1)  # adjust space between Axes

# plot the same data on both Axes
for tau in key_ttm:
    ax2.plot(df_forwards.index, df_forwards[tau], color=cm(dict_key_ttm_clrs[tau]), label=dict_key_ttm_labels[tau])
df_hhng_spot.plot(ax=ax2, legend=False, color='black', linestyle='dotted')
df_hhng_spot.plot(ax=ax1, color='black', linestyle='dotted', legend=False)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(10, 25)  # outliers only
ax2.set_ylim(0, 10)  # most of the data

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
ax2.axhline(y=0, color=colors[5], label='_nolegend_')
ax2.set_ylabel('$ / MMBTU')
ax2.set_xlabel('')
ax2.set_xlim(timeline.min(), timeline.max())
ax1.legend(['NG-HH Spot'])
ax2.legend(key_ttm_labels, title='NG-HH Futures', ncols=2, title_fontsize=25)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plotfolder, f'NG_HH_futures.pdf'))

weeks = np.arange(52)/52
list_dfs = [df_us_ng_storage, df_us_ng_storage_net_change]
fig, ax = plt.subplots(ncols=2, figsize=(20,10))
axs = ax.flatten()
for i in range(2):
    if i == 1:
        axs[i].axhline(y=0, color=colors[5], label='_nolegend_')
    axs[i].axvline(x=0, color=colors[5], label='_nolegend_')
    lb = list_dfs[i].Low.values
    ub = list_dfs[i].High.values
    avg = list_dfs[i].Avg.values
    this_year = list_dfs[i]['2024'].values

    axs[i].fill_between(weeks, lb, ub, color=colors[0], alpha=.5, label='2019-2023 Range')
    axs[i].plot(weeks, avg, color=colors[0], label='2019-2023 Average')
    axs[i].plot(weeks, this_year, color=colors[2], label='2024')

    axs[i].set_xlim(0,1)
    axs[i].set_xticks(np.arange(12) / 12, labels=months, rotation=45)

axs[0].legend()
axs[0].set_title('Working gas in underground storage', size=25)
axs[1].set_title('Net change in working gas in underground storage', size=25)
axs[0].set_ylabel('Billion cubic feet')
axs[1].set_ylabel('Billion cubic feet')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plotfolder, f'NG_storage_levels.pdf'))


t = pd.to_datetime('20210726')
loc = np.argwhere(timeline == t).item(0)

fig, ax = plt.subplots(figsize=(9,9), subplot_kw=dict(projection="3d"))
tmp = df_vols_cube.loc[t].iloc[:, 1:].dropna(axis=1)
tau = tmp.columns.values
Y = tmp.values **.5
m, n = Y.shape
X = np.ones((m,n)) * x_grid.reshape(m, 1)
TAU = np.ones((m,n)) * tau.reshape(1, n)
ax.scatter(X.flatten(), TAU.flatten(), Y.flatten() *100, color=cm(Y.flatten()), label=r'$I_t$', s=50)
ax.plot_wireframe(X, TAU, Y*100,  label=r'$\widehat{I}_{t}$', linewidth=1/2, color='black')
ax.view_init(azim=45, elev=25)
# ax.set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25, pad=0)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'Implied Volatility $I(x, \tau)$ (%)', labelpad=10, rotation=90)
ax.set_ylabel(r'Time-to-Maturity $\tau$ (years)',labelpad=20)
ax.set_xlabel(r'Monyeness $x=\log K/F$', labelpad=20)
plt.suptitle(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=23, y=0.9)
plt.tight_layout()
# plt.show()

plt.savefig(os.path.join(plotfolder, f'NG_surface_a.pdf'))

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from  matplotlib.cm import ScalarMappable as cmsm
from matplotlib.colors import Normalize
fig, ax = plt.subplots(figsize=(10,9))
tmp = df_vols_cube.loc[t].iloc[:, 1:].dropna(axis=1)
tau = tmp.columns.values
Y = tmp.values.astype(float)
m, n = Y.shape
X = np.ones((m,n)) * x_grid.reshape(m, 1)
X = X.flatten().astype(float)

for col in reversed(range(n)):
    ax.scatter(x_grid, Y[:, col] *tau[col] *100, color=cm(tau[col]/2.25), vmin=0, vmax=2, cmap=cm, alpha=0.9)
cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
cb = fig.colorbar(cmsm(cmap=cm, norm=Normalize(0, 2.25)), cax=cax)
cb.set_label(r'$\tau$', rotation=0, labelpad=10)
ax.set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
ax.set_ylabel(r'Total Variances  $I^2(x, \tau) \times \tau$ (%)')
ax.set_xlabel(r'Monyeness $x=\log K/F$')
plt.tight_layout()
plt.savefig(os.path.join(plotfolder, f'NG_surface_b.pdf'))


from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from  matplotlib.cm import ScalarMappable as cmsm
from matplotlib.colors import Normalize
fig, ax = plt.subplots(figsize=(10,9))
tmp = (df_vols_cube).groupby(by='CLOSEST_MONEYNESS').mean() **.5
tau = tmp.columns.values
Y = tmp.values.astype(float)
m, n = Y.shape
X = np.ones((m,n)) * x_grid.reshape(m, 1)
X = X.astype(float)
TAU = np.ones((m,n)) * tau.reshape(1, n)
cf = ax.contourf(X, Y*100, TAU.astype(float), cmap=cm,
                               levels=np.linspace(tau_grid.min(), tau_grid.max(), 50), alpha=.5)
cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
cb = fig.colorbar(cmsm(cmap=cm, norm=Normalize(0, 2.25)), cax=cax)
cb.set_label(r'$\tau$', rotation=0, labelpad=10)
ax.set_title(r'Historical Average', size=25)
ax.set_ylabel(r'Avg. Implied Volatility  $I(x, \tau)$ (%)')
ax.set_xlabel(r'Monyeness $x=\log K/F$')
plt.tight_layout()
plt.savefig(os.path.join(plotfolder, f'NG_surface_c.pdf'))


df_data = df_vols_cube.copy()
df_data.iloc[:, 1:] **= .5
x_bins = dict({-0.4: 'DITM', -0.3: 'DITM',
               -0.2: 'ITM', -0.1: 'ITM',
                -0.05: 'ATM', -0.025: 'ATM',  0.0:'ATM',  0.025:'ATM', 0.05:'ATM',
                0.1:'OTM',  0.2: 'OTM',
               0.3:'DOTM', 0.4:'DOTM'})
tau_bins = pd.cut(tau_grid,  bins=[0, 0.0833, 0.25, 0.5, 1, 2.5]).unique().tolist()
x_bins_labels = ['DITM', 'ITM', 'ATM', 'OTM', 'DOTM']

df_data = df_data.reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS'])
df_data.columns = pd.cut(df_data.columns,  bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
df_data = df_data.reset_index().set_index(['DATE'])
df_data[['CLOSEST_MONEYNESS']] = df_data[['CLOSEST_MONEYNESS']].replace(x_bins).astype(str)


df_nobs = pd.DataFrame()
df_avg = pd.DataFrame()
df_std = pd.DataFrame()
for tau in tau_bins:
    for x in x_bins_labels:
        df_bucket = df_data[df_data.CLOSEST_MONEYNESS == x][tau].values.flatten()
        df_nobs.loc[x, tau] = np.invert(np.isnan(df_bucket)).sum()
        df_avg.loc[x, tau] = np.nanmean(df_bucket *HUNDRED)
        df_std.loc[x, tau] = np.nanstd(df_bucket * HUNDRED)
        # df_avg = df_bucket.groupby(by='DATE').mean()
        # df = pd.concat([df, df_avg_errors], axis=1)

df_summary_stats = pd.concat([df_nobs, df_avg,df_std], axis=1)
df_summary_stats.style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, 'iv_summary_stats.tex'))