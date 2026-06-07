import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from config_utils.config_utils_main import import_ts_ivs
from numba.typed import List
from joblib import load
from measurement_equations.carr_wu_seasonal import seasonal_effect
from measurement_equations.carr_wu_seasonal import map_moments
from calculator.forward_curve_calculator import ForwardCurveCalculator
from calculator.yield_curve_calculator import ZeroRatesCalculator
from pricers.variance_swap_rate import VarianceSwapPricer
from models.regression.least_squares import OLS
from models.regression.rolling_window_least_squares import RollingWindowLeastSquareAnalysis as RWLSA


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
save_fig = True
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
key_ttm = [0.0833, 0.25, 0.5, 1, 1.5, 2]
dict_key_ttm_labels = dict({0.0833: '1M', 0.25: '3M', 0.5: '6M', 1: '1Y', 1.5: '18M', 2: '2Y'})
dict_key_ttm_clrs = dict({0.0833: 0.0833/2.6, 0.25: 0.5/2.6, 0.5: 1/2.6, 1: 1.5/2.6, 1.5: 2/2.6, 2: 2.5/2.6})
key_ttm_labels = ['1M', '3M', '6M', '1Y', '18M', '2Y']
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
df_hhng_spot = df_hhng_spot.set_index(pd.to_datetime(df_hhng_spot.DATE), drop=True).dropna()[['DHHNGSP']].reindex(timeline)
df_hhng_spot = df_hhng_spot.astype(float).interpolate()
df_vix = pd.read_csv(os.path.join(datafolder, 'indices', 'VIX.csv'))
df_vix = df_vix.set_index(pd.to_datetime(df_vix.DATE), drop=True)[['VIX']].dropna().reindex(timeline)
df_vix = df_vix.astype(float).interpolate()


df_ng_snc = pd.read_excel(os.path.join(datafolder, 'eia', 'ngshistory.xls'),
                          sheet_name='weekly_net_changes',  skiprows=6
                          )[['Week ending', 'Total Lower 48']].rename(columns={'Week ending':'DATE', 'Total Lower 48':'STORAGE'})

df_ng_snc = df_ng_snc.set_index(pd.to_datetime(df_ng_snc.DATE) - pd.offsets.BusinessDay(n=1))[['STORAGE']].dropna().reindex(timeline).ffill().bfill()


fcc = ForwardCurveCalculator(os.path.join(datafolder, 'comdty_forwards'))
zrc = ZeroRatesCalculator(os.path.join(datafolder, 'rates'))

df_forwards = fcc.interpolate(tau_grid, timeline)
df_rates = zrc.interpolate(tau_grid, timeline)

vsp = VarianceSwapPricer(df_vols_cube, df_forwards, df_rates)
vsp.surface_interpolator()
vsp.compute_variance_swap_rates()

in_sample = (timeline <= train_end).sum()
tmp = df_vols_cube.set_index('CLOSEST_MONEYNESS')
tmp.isna().groupby(by=tmp.index).mean() * 100



## MODEL TO ANALYZE
mean = 'carr_wu_disprop2_s4'
variance = 'mvn-spline'
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

model.compute_stats()



labels = ['2W', '1M', '3M', '6M', '1Y', '18M', '2Y']
tau_to_plot = [0.0417, 0.0833, 0.25, 0.5, 1, 1.5, 2]
df_atm = model.df_vols[model.df_vols.CLOSEST_MONEYNESS == 0.0].iloc[:,1 :]
df_atm_ws = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0.0].iloc[:,1 :]
df_atm_wos = model.df_fitted_deseasonalized_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0.0].iloc[:,1 :]

fig, ax = plt.subplots(figsize=(20,15), nrows=3, sharex=True)
for tau in tau_grid:
    ax[0].scatter(model._time_line, df_atm[tau].values * HUNDRED, s=10, alpha=0.1, color=cm(tau / 2.2), label='_nolegend_')
i = 0
for tau in tau_to_plot:
    ax[0].plot(model._time_line, df_atm_ws[tau].values * HUNDRED, color=cm(tau/2.2),  label=labels[i])
    ax[1].plot(model._time_line, df_atm_wos[tau].values * HUNDRED,  color=cm(tau/2.2), label='_nolegend_')
    ax[2].plot(model._time_line, (df_atm_ws[tau].values-df_atm_wos[tau].values) * HUNDRED,  color=cm(tau/2.2))
    ax[0].set_xlim(model._time_line.min(), model._time_line.max())
    ax[1].set_xlim(model._time_line.min(), model._time_line.max())
    ax[2].set_xlim(model._time_line.min(), model._time_line.max())
    i +=1
ax[1].plot(model._time_line, np.exp(model.a_t[:, 2])*HUNDRED, color='black', label=r'$\nu_t$', linestyle='dotted')
ax[0].legend(ncol=2, loc='upper left')
ax[1].legend()
ax[0].set_title('Fitted ATM Term-Structure', size=30)
ax[1].set_title('Regular Component', size=30)
ax[2].set_title('Seasonal Component', size=30)
ax[0].set_ylabel(r'$\widehat{I}_{t|t}(0, \tau)$ (%)')
ax[1].set_ylabel(r'$\widehat{I}^R_{t|t}(0, \tau)$ (%)')
ax[2].set_ylabel(r'$\widehat{I}^S_{t|t}(0, \tau)$ (%)')

for i in range(3):
    ax[i].axvline(x=pd.to_datetime(train_end), color=colors[5], label="_nolegend_", linestyle='dotted')
    if i == 0:
        ax[i].text(x=pd.to_datetime('20230320'), y=120, s=r'In-sample', size=18)
        ax[i].text(x=pd.to_datetime('20230901'), y=120, s=r'Out-of-sample', size=18)

plt.tight_layout()
plt.savefig(os.path.join(plotfolder, f'atm_fit_{variance}.pdf'))
# plt.show()


### PLOT AVERAGE SEASONAL EFFECTS
f_t = model.a_t.mean(axis=0)
P_t = model.P_t.mean(axis=0)
M = 500
tau = np.arange(1, 365*2+1).reshape(365*2, 1) /365
theta_sim = np.zeros((len(tau), M))
theta_sim_decay = np.zeros((len(tau), M)).copy()
FIXED = List()
FIXED.append(tau*0)
FIXED.append(tau)
FIXED.append(tau)
FIXED.append(tau*0)
for i in range(500):
    f_i = np.random.multivariate_normal(f_t.flatten(), P_t)
    I0 = cw_atm(f_i, tau, add_S=True)   #w S-d
    I1 = cw_atm(f_i, tau, add_S=False)   # w S-d

    deltas = f_i[-8:]
    theta_sim[:, i] = seasonal_effect(deltas, tau).flatten()
    theta_sim_decay[:, i] = (I0 - I1).flatten()


fig, ax = plt.subplots(figsize=(20,10), ncols=2, sharey=False, width_ratios=[4, 6])
axs = ax.flatten()
list_thetas = [theta_sim, theta_sim_decay]
for i in range(2):
    axs[i].axhline(y=0, color=colors[5], label='_nolegend_')
    axs[i].axvline(x=0, color=colors[5], label='_nolegend_')
    lb = np.quantile(list_thetas[i], 0.025, axis=1) * HUNDRED
    ub = np.quantile(list_thetas[i], 0.975, axis=1) * HUNDRED
    avg = np.mean(list_thetas[i], axis=1) * HUNDRED
    n = len(avg)
    axs[i].plot(tau.flatten(), avg, color=colors[i], label='CI(95\%)')
    axs[i].fill_between(tau.flatten(), lb, ub, color=colors[i], alpha=.5, label='CI(95\%)')

ax_twinx = axs[0].twinx()
ax_twinx.plot(np.arange(52)/52, df_us_ng_storage_net_change.Avg.values, color=colors[2], label=r'$\Delta$Storage')
ax_twinx.set_ylabel('Billion cubic feet')
ax_twinx.set_ylim(-200, 200)
axs[0].set_xlim(0,1)
axs[1].set_xlim(0,2)
axs[0].set_ylim(-200, 200)
axs[0].set_title('Average Seasonality', size=25)
axs[1].set_title('Average Maturity Adjusted Seasonality', size=25)
axs[1].axvline(x=1, color=colors[5], label='_nolegend_', linestyle='dotted')
axs[0].set_xticks(np.arange(12)/12, labels=months, rotation=45)
axs[1].set_xticks(np.arange(24)/12, labels=months*2, rotation=45)
axs[0].set_ylabel('%')
axs[1].set_ylabel('%')
axs[0].set_xlabel(r'$T=t+\tau$  $(t=0)$')
axs[1].set_xlabel(r'$T=t+\tau$  $(t=0)$')
axs[0].legend([r'$\overline{\theta}_t(\tau) = X_t(\tau)\overline{\delta}_t$', 'CI(95%)'])
axs[1].legend([r'$\overline{\theta}_t(\tau) = X_t(\tau)\overline{\delta}_t \times e^{\overline{\eta}_{2} \times \tau}$','CI(95%)'])
ax_twinx.legend(loc='lower right')
axs[0].text(x=0, y=-10, s=r'$T =0$', size=20)
axs[0].text(x=21.5/24, y=-10, s=r'$T=1$', size=20)
axs[1].text(x=0, y=-1, s=r'$T =0$', size=20)
axs[1].text(x=1, y=-1, s=r'$T=1$', size=20)
axs[1].text(x=22.5/12, y=-1, s=r'$T=2$', size=20)
plt.tight_layout()
if save_fig:
    plt.savefig(os.path.join(plotfolder, f'HH_avg_seasonality.pdf'))
else:
    plt.show()


### PLOT AVERAGE FORWARD LOOKING SEASONAL EFFECTS
fit_ws = model.df_fitted_vols.groupby(by='CLOSEST_MONEYNESS').mean().values
fit_wos = model.df_fitted_deseasonalized_vols.groupby(by='CLOSEST_MONEYNESS').mean().values
Y = model.df_vols.groupby(by='CLOSEST_MONEYNESS').mean().values

ttm = np.arange(12) /12
df_atm = model.df_vols[model.df_vols.CLOSEST_MONEYNESS == 0].iloc[:, 1:] **2
df_atm_ws = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0].iloc[:, 1:] **2
df_atm_wos = model.df_fitted_deseasonalized_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0].iloc[:, 1:] **2

df_seasonal_premia = df_atm_ws - df_atm_wos
df_avg_seasonal_premia = df_seasonal_premia.groupby(by=df_seasonal_premia.index.month).mean() * HUNDRED
df_lb_seasonal_premia = df_seasonal_premia.groupby(by=df_seasonal_premia.index.month).quantile(0.16)
df_ub_seasonal_premia = df_seasonal_premia.groupby(by=df_seasonal_premia.index.month).quantile(1-0.16)

fig, ax = plt.subplots(figsize=(20,10), ncols=3, nrows=2,  sharey='row')
axs = ax.flatten()
i = 0
for tau in key_ttm:
    axs[i].axhline(y=0, color=colors[5], label='_nolegend_')
    axs[i].plot(ttm, df_avg_seasonal_premia[tau].values, label='_nolegend_')
    for m in range(12):
        axs[i].scatter(ttm[m], df_avg_seasonal_premia[tau].values[m],
                       alpha=.6, color=colors[2], edgecolors=colors[2], s=100, linewidths=10)
        T = np.ceil(ttm[m]*12 + tau *12).astype(int)
        axs[i].annotate((months*3)[T], xy=(ttm[m], df_avg_seasonal_premia[tau].values[m] + 1), rotation=45)
    axs[i].set_xticks(ttm, labels=months, rotation=45)
    axs[i].set_xlim(-0.01, 1)
    axs[i].set_title(r'$\overline{\theta}_t$(' + dict_key_ttm_labels[tau] + ')', size=30)
    i +=1

for i in range(3, 6):
    axs[i].set_xlabel(r'time $t=0$')
axs[0].set_ylim(-10, 40)
axs[3].set_ylim(-5, 15)
axs[0].set_ylabel('%')
axs[3].set_ylabel('%')
plt.tight_layout()
if save_fig:
    plt.savefig(os.path.join(plotfolder, f'HH_avg_forward_seasonality.pdf'))
else:
    plt.show()



x = 0
Ix = vsp.variance_surfaces[vsp.variance_surfaces.CLOSEST_MONEYNESS == 0].iloc[:, 1:] **.5  * HUNDRED
Ix_hat = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0].iloc[:, 1:] * HUNDRED
Ix_regular_hat = model.df_fitted_deseasonalized_vols[model.df_fitted_deseasonalized_vols.CLOSEST_MONEYNESS == 0].iloc[:, 1:] * HUNDRED
Ix_seasonal_hat = Ix_hat - Ix_regular_hat
noise = pd.DataFrame(model.rmse_t, index=timeline, columns=["NOISE"])
df_rv_p = (np.log(df_forwards).diff()**2).ewm(alpha=1-0.94).mean().bfill().ffill()**.5 * HUNDRED
df_rv_q = pd.DataFrame((np.exp(model.a_t[:, 2])**2/365 *1)**.5 * HUNDRED, index=timeline)
df_vrp = - (df_rv_p - df_rv_q.values)

for h in [0]:
    list_results = []
    order = ['Const', 'Const_se', 'RS', 'RS_se', 'R', 'R_se', 'S', 'S_se',
             # 'STORAGE',  'STORAGE_se',
             # 'WED', 'WED_se', 'THU', 'THU_se',
             # 'RV', 'RV_se',
             # 'VIX', 'VIX_se',
             # "NOISE",  "NOISE_se",
             'AdjR2']
    for tau in key_ttm:
        y = Ix[[tau]].diff().dropna()
        idx = y.index
        n = y.shape[0]
        ones = pd.DataFrame(np.ones_like(y), columns=['Const'], index=y.index).loc[idx]
        RS = Ix_hat[[tau]].diff().dropna().rename(columns={tau:'RS'}).loc[idx]
        R = Ix_regular_hat[[tau]].diff().dropna().rename(columns={tau:'R'}).loc[idx]
        S = Ix_seasonal_hat[[tau]].diff().dropna().rename(columns={tau:'S'}).loc[idx]
        STRG = df_ng_snc.copy().loc[idx]
        WEEKDAYS = pd.get_dummies(pd.DataFrame(y.index.dayofweek.tolist(), index=y.index, columns=['Day']).astype(str)).astype(float)[['Day_2', 'Day_3']]
        WEEKDAYS.columns = ['WED', 'THU']
        RV = df_vrp[[tau]].diff().dropna().loc[idx].rename(columns={tau:'RV'})
        VIX = df_vix.diff().dropna().loc[idx]
        NOISE = noise.diff().dropna().loc[idx]

        X0 = pd.concat([ones, RS], axis=1)
        X1 = pd.concat([ones, R], axis=1)
        X2 = pd.concat([ones, R, S], axis=1)
        beta0 = np.array([[.0], [1.], [1.]])
        # X3 = pd.concat([ones, STRG/100], axis=1)
        # X4 = pd.concat([ones, WEEKDAYS], axis=1)
        # X5 = pd.concat([ones, STRG/100, WEEKDAYS], axis=1)
        # X6 = pd.concat([ones, WEEKDAYS], axis=1)
        # X7 = pd.concat([ones, NOISE], axis=1)
        # X8 = pd.concat([ones, R, S, STRG, WEEKDAYS], axis=1)

        list_X = [X0, X1, X2,
                  # X3, X4, X5, X8
                  # X6, X7, X8
                  ]
        df_results = pd.DataFrame()
        for X in list_X:
            ols_model = OLS(y.values.reshape(n, 1)[h:], X.values[:n-h], lag_h=5)
            ols_model.fit()
            ols_model.summary(idx=X.columns, beta0=beta0[:X.shape[1]])
            df_tmp = ols_model._summary.copy().reset_index()
            i = 0
            for j in np.arange(1, len(X.columns)*2, 2):
                df_tmp.iloc[j, 0] = f'{X.columns[i]}_se'
                i += 1

            df_results = pd.concat([df_results, df_tmp.set_index('index')], axis=1)

        list_results.append(df_results.loc[order])


    df_results_data = pd.concat([pd.concat(list_results[:3], axis=1), pd.concat(list_results[3:], axis=1)], axis=0)
    df_results_data.to_latex(os.path.join(tablefolder, f'seasonal_variation_data_{mean}_{variance}_h={h}.tex'))


df_rv_p = (np.log(df_forwards).diff()**2).ewm(alpha=1-0.94).mean()**.5 * HUNDRED
df_rv_q = pd.DataFrame((np.exp(model.a_t[:, 2])**2/365 *1)**.5 * HUNDRED, index=timeline)
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df_rv_p.iloc[:, :2].mean(axis=1))
ax.plot(df_rv_q)
plt.tight_layout()
plt.show()




# df_results_data = pd.concat([pd.concat(list_results[:3], axis=1), pd.concat(list_results[3:], axis=1)], axis=0)
# df_results_data.to_latex(os.path.join(tablefolder, f'seasonal_variation_data_{mean}_{variance}.tex'))
#
# noise = pd.DataFrame(np.log(model.rmse), index=timeline, columns=["NOISE"])
# list_results = []
# order = ['Const','Ix_seas_hat', 'AdjR2']
# for tau in key_ttm:
#     y = Ix_hat[[tau]].diff().dropna()
#     n = y.shape[0]
#     ones = pd.DataFrame(np.ones_like(y), columns=['Const'], index=y.index)
#     tmp2 = Ix_seasonal_hat[[tau]].diff().dropna().rename(columns={tau:'Ix_seas_hat'})
#
#     X1 = pd.concat([ones, tmp2], axis=1)
#
#     list_X = [X1]
#     df_results = pd.DataFrame()
#     for X in list_X:
#         ols_model = OLS(y.values.reshape(n, 1), X.values)
#         ols_model.fit()
#         ols_model.summary()
#         df_summary = ols_model._summary
#         cols = X.columns.tolist() + ["AdjR2"]
#         df_summary.columns = cols
#
#         df_results = pd.concat([df_results, df_summary], axis=0)
#
#     list_results.append(df_results[order])
#
# df_results_fitted = pd.concat(list_results, axis=0)
# df_results_fitted.to_latex(os.path.join(tablefolder, f'seasonal_variation_data_fitted_{mean}_{variance}.tex'))
#
#
#
# noise = pd.DataFrame((model.mae), index=timeline, columns=["NOISE"])
# list_results = []
# order = ['Const', 'Day_2', 'Day_3','NOISE', 'AdjR2']
# for tau in key_ttm:
#     y = Ix[[tau]].diff().dropna()
#     n = y.shape[0]
#     ones = pd.DataFrame(np.ones_like(y), columns=['Const'], index=y.index)
#     tmp1 = pd.get_dummies(pd.DataFrame(y.index.dayofweek.tolist(), index=y.index, columns=['Day']).astype(str)).astype(float).drop(columns=['Day_0', 'Day_1', 'Day_4'])
#     tmp2 = noise.diff().dropna()
#
#     X1 = pd.concat([ones, tmp1], axis=1)
#     X2 = pd.concat([ones, tmp2], axis=1)
#     X3 = pd.concat([ones, tmp1, tmp2], axis=1)
#
#     list_X = [X1, X2, X3]
#     df_results = pd.DataFrame()
#     for X in list_X:
#         ols_model = OLS(y.values.reshape(n, 1), X.values)
#         ols_model.fit()
#         ols_model.summary()
#         df_summary = ols_model._summary
#         cols = X.columns.tolist() + ["AdjR2"]
#         df_summary.columns = cols
#
#         df_results = pd.concat([df_results, df_summary], axis=0)
#
#     list_results.append(df_results[order])
#
# df_results_data = pd.concat([pd.concat(list_results[:3], axis=1), pd.concat(list_results[3:], axis=1)], axis=0)
#
#
#
# def add_signficance_sign(t):
#     if (np.abs(t) >= 1.645 and np.abs(t) < 1.96):
#         return '$^{*}$'
#     elif (np.abs(t) >= 1.96 and np.abs(t) < 2.576):
#         return '$^{**}$'
#     elif np.abs(t) >= 2.576:
#         return '$^{***}$'
#     else:
#         return ''
#
# df_fitted_atm_vols = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0].iloc[:, 1:].copy() * HUNDRED
# for tau in key_ttm:
#     print(tau)
#     y = df_fitted_atm_vols[[tau]].diff().dropna() * HUNDRED
#     X = pd.get_dummies(pd.DataFrame(y.index.dayofweek.tolist(), index=y.index, columns=['Day']).astype(str)).astype(float)
#     n = y.shape[0]
#     ols_model = OLS(y.values.reshape(n,1), X.values)
#     ols_model.fit()
#     t_val = pd.DataFrame(ols_model._beta / ols_model._se)
#     print(np.concatenate([ols_model._beta, ols_model._se], axis=1))
#     print(t_val.map(add_signficance_sign))
#     print(ols_model._r2 *HUNDRED)

# pd.concat([df_fitted_atm_vols[key_ttm], df_vix], axis=1).diff().corr()

# df_returns_spot = (np.log(df_hhng_spot).diff() * 100)
# df_rv_spot = df_returns_spot.clip(df_returns_spot.quantile(0.005), df_returns_spot.quantile(0.995), axis=1).rolling(21).var()
# df_rv_spot.plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
#
# df_returns_forwards = (np.log(df_forwards).diff()[[0.0833, 0.25, 0.5, 1, 1.5, 2]]*100)
# df_rv_forwards = (df_returns_forwards.clip(df_returns_forwards.quantile(0.025), df_returns_forwards.quantile(0.975), axis=1)**2).ewm(alpha=0.04).mean()
# df_rv_forwards.plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
#
# (vsp.variance_swap_rates[[0.0833, 0.25, 0.5, 1, 1.5, 2]]).plot(figsize=(20,10)), plt.tight_layout(), plt.show()
#
# (np.log(df_forwards).diff()[[0.0833, 0.25, 0.5, 1, 1.5, ]]*100).rolling(21).var().plot(figsize=(20,10),legend=False), plt.tight_layout(), plt.show()
#
# nu = pd.DataFrame(np.exp(model.a_t[:, 2]), index=timeline)
# (nu**2 *100).plot(figsize=(20,10)), plt.tight_layout(), plt.show()
#
#
# from arch.univariate import ConstantMean, EGARCH, SkewStudent, FIGARCH
# for tau in [0.0833, 0.25, 0.5, 1, 1.5, 2]:
#     fig, ax = plt.subplots(figsize=(20,10), nrows=2)
#     tmp = df_returns_forwards[[tau]].dropna()
#     tmp = tmp.clip(tmp.quantile(0.001), tmp.quantile(1 - 0.001), axis=1)
#     if tau == 2:
#         tmp = tmp.clip(tmp.quantile(0.05), tmp.quantile(1 - 0.05), axis=1)
#
#     garch = ConstantMean(tmp)
#     garch.volatility = EGARCH()
#     garch.distribution = SkewStudent()
#     res = garch.fit()
#     y1 =(vsp.variance_swap_rates[[tau]]*100)
#     y2 = (res.conditional_volatility**2)
#     y1.plot(ax=ax[0])
#     y2.plot(ax=ax[0])
#     basis = y1.values.flatten().astype(float)[1:] - y2.values.astype(float)
#     print(np.median(basis))
#     pd.DataFrame(basis, index=y1.index[1:]).plot(ax=ax[1])
#     plt.tight_layout()
#     plt.show()
#
#
# slope = df_forwards[1] - df_forwards[0.0833]
# fig, ax = plt.subplots(figsize=(20,10))
# ax_twin = ax.twinx()
# slope.plot(ax=ax)
# # pd.DataFrame(model.a_t[:, 0], index=timeline).plot(ax=ax_twin, color=colors[2])
# pd.DataFrame(model.a_t[:, 3], index=timeline).plot(ax=ax_twin, color=colors[2])
# plt.tight_layout(), plt.show()
#
#
#
#
#
