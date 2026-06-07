import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from config_utils.config_utils_main import import_ts_ivs
from measurement_equations.carr_wu_seasonal import map_moments
from measurement_equations.carr_wu_seasonal import seasonal_effect
from calculator.forward_curve_calculator import ForwardCurveCalculator
from calculator.yield_curve_calculator import ZeroRatesCalculator
from models.static.surface_models.seasonal_ssvi_model import SeasonalSurfaceSVIModel
from numba.typed import List
from joblib import dump, load
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import interp1d
from copy import deepcopy
from statistical_tests.model_confidence_set_test import ModelConfidenceSet as MCS

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
key_ttm = [0.0833, 0.25, 0.5, 1, 1.5, 2]
dict_key_ttm_labels = dict({0.0833: '1M', 0.25: '3M', 0.5: '6M', 1: '1Y', 1.5: '18M', 2: '2Y'})
dict_key_ttm_clrs = dict({0.0833: 0.0833/2.6, 0.25: 0.5/2.6, 0.5: 1/2.6, 1: 1.5/2.6, 1.5: 2/2.6, 2: 2.5/2.6})
key_ttm_labels = ['1M', '3M', '6M', '1Y', '18M', '2Y']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

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
in_sample = ((timeline <= train_end).sum()).astype(int)
means = [
    'carr_wu_standard',
    'carr_wu_prop_fixed',
    'carr_wu_s1_fixed', 'carr_wu_s2_fixed',
    'carr_wu_prop_s1_dyn_fixed', 'carr_wu_prop_s2_dyn_fixed',
    'carr_wu_prop_s1', 'carr_wu_prop_s2',
    'carr_wu_disprop_s1_dyn_fixed', 'carr_wu_disprop_s2_dyn_fixed',
    'carr_wu_disprop_s1', 'carr_wu_disprop_s2','carr_wu_disprop_s3','carr_wu_disprop_s4',
    'carr_wu_disprop2_s3', 'carr_wu_disprop2_s4',
]

variances = ['iid', 'mvn-spline']
# cols = ['k', 'iid-params', 'iid-ll', 'iid-rmspe-in', 'iid-rmspe-out',
#         'mvn-spline-params', 'mvn-spline-ll', 'mvn-spline-rmspe-in', 'mvn-spline-rmspe-out']
# horizons = [1, 5, 21, 63, 126, 252]
# df_results = pd.DataFrame(index=means, columns=cols)
# ### FIT MODELS
# for mean in means:
#     print(mean)
#     for variance in variances:
#         print(variance)
#
#         model_name = f'{mean}_{variance}.pkl'
#         dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
#         model = DynamicSurfaceModels(df_surface=df_vols_cube)
#         model.specify_measurement_equation(mean=mean, variance=variance,
#                                            moneyness_pillars=x_knots, maturity_pillars=tau_knots,
#                                            cyclical_knots=cyclical_knots
#                                            )
#         model._set_model_params(*dict_results['set_params'])
#
#         try:
#             model.run_filter()
#         except:
#             model.run_filter()
#
#         model.compute_stats(N_fit=in_sample)
#         model.forecast(horizons=horizons)
#         # pd.DataFrame(model.a[:, :, 0], index=model._time_line).plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
#         # pd.DataFrame(model.rmspe, index=model._time_line).plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
#         pd.DataFrame(model.r2, index=model._time_line).plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
#         # print(model._mle._mle_optim_results)
#         # print(np.diag(model.Q))
#         # print(np.diag(model.H).min())
#         # print(np.diag(model.H).max())
#
#         print(pd.DataFrame(model.r2, index=model._time_line).mean())
#         print('\n')
#         #
#         try:
#             fig, ax = plt.subplots(figsize=(20,10), ncols=2, sharex=True)
#             ax[0].plot(tau_grid, model._v_tau(tau_grid))
#             ax[1].plot(x_grid, model._u_x(x_grid))
#             ax[0].scatter(model._v_tau.x, model._v_tau.y)
#             ax[1].scatter(model._u_x.x, model._u_x.y)
#             plt.suptitle(f'{mean}_{variance}', size=25)
#             plt.tight_layout()
#             plt.show()
#         except:
#             pass
#
#         mspe = (model.rmspe.copy() /100) **2
#         df_mspe = pd.DataFrame(mspe, index=model._time_line, columns=['Avg.'])
#         p = np.array([5, 25, 50, 75, 95]) /100
#         stats = pd.concat([pd.DataFrame(model.rmse), pd.DataFrame(model.rmspe), pd.DataFrame(model.mae),
#                            pd.DataFrame(model.mape), pd.DataFrame(model.r2)], axis=1)
#         stats.columns = ['RMSE', 'RMSPE', 'MAE', 'MAPE', 'R2']
#         stats_summary = stats.iloc[10:].describe(percentiles=p).T
#         # pd.concat([stats.iloc[10:].describe(percentiles=p).T,
#         #                    stats.iloc[in_sample:].describe(percentiles=p).T], axis=0)
#         stats_summary = stats_summary.drop(columns=['count'])
#         ll = model.LL[10:in_sample].sum().round().astype(int)
#         k = model._k_params
#         aic = 2*k - 2*ll
#         print(stats_summary)
#         df_results.loc[mean, 'k'] = model._k
#         df_results.loc[mean, f'{variance}-params'] = k
#         df_results.loc[mean, f'{variance}-ll'] = ll
#         df_results.loc[mean, f'{variance}-aic'] = aic
#         df_results.loc[mean, f'{variance}-rmspe-in'] = mspe[10:in_sample].mean() **.5  * HUNDRED
#         df_results.loc[mean, f'{variance}-rmspe-out'] = mspe[in_sample:].mean() **.5  * HUNDRED
#
#         dump(model.df_squared_residuals_vols, os.path.join(picklefolder, 'fitted_models', ticker, 'squared_errors', model_name))
#         dump(model.df_residuals_vols, os.path.join(picklefolder, 'fitted_models', ticker, 'errors', model_name))
#         dump(model.df_rel_squared_residuals_vols,os.path.join(picklefolder, 'fitted_models', ticker, 'squared_relative_errors', model_name))
#         dump(model.df_fitted_vols,os.path.join(picklefolder, 'fitted_models', ticker, 'fitted_vols', model_name))
#         dump(df_mspe, os.path.join(picklefolder, 'fitted_models', ticker, 'daily_avg_squared_relative_errors', model_name))
#         dump(stats_summary, os.path.join(picklefolder, 'fitted_models', ticker, 'statistics', model_name))
#
#
#         forecast_objects = [model._dict_rmspe, model._dict_rel_squared_residuals_vols]
#         dump(forecast_objects, os.path.join(picklefolder, 'fitted_models', ticker, 'forecast_errors', model_name))
#
#
# cols = ['k', 'iid-params', 'iid-ll', 'iid-aic', 'iid-rmspe-in', 'iid-rmspe-out',
#        'mvn-spline-params', 'mvn-spline-ll', 'mvn-spline-aic', 'mvn-spline-rmspe-in',
#        'mvn-spline-rmspe-out',]
# df_results[cols].style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, 'results_fitted_models.tex'))


def cell_color(x):
    if x >= 0.1:
        c =  "\cellcolor[HTML]{96FFFB}"
    elif x >= 0.05 and x < 0.1:
        c = "\cellcolor[HTML]{FFFC9E}"
    elif x >= 0.01 and x < 0.05:
        c = "\cellcolor[HTML]{FFCCC9}"
    else:
        c = ""

    return c

models_iid = [f'{x}_iid' for x in means]
models_mxn = [f'{x}_mvn-spline' for x in means]
h = 1
df_mspe_train = pd.DataFrame()
df_mspe_test = pd.DataFrame()
df_rmspe = pd.DataFrame()
for variance in variances:
    df_tmp = pd.DataFrame()
    for mean in means:
        model_name = f'{mean}_{variance}'
        forecast_objects = load(os.path.join(picklefolder, 'fitted_models', ticker, 'forecast_errors', f'{model_name}.pkl'))
        rmspe, df_errors = forecast_objects[0][h], forecast_objects[1][h]
        df_mspe_train[model_name] = rmspe[10:in_sample] ** 2
        df_mspe_test[model_name] = rmspe[in_sample:] ** 2
        df_rmspe.loc[mean, f'{variance}-rmspe-in'] = (rmspe[10:in_sample] ** 2).mean()
        df_rmspe.loc[mean, f'{variance}-rmspe-out'] = (rmspe[in_sample:] ** 2).mean()

cols = ['iid-rmspe-in', 'iid-rmspe-out', ' mvn-spline-rmspe-in', 'mvn-spline-rmspe-out', ]
alpha, w, B = 0.05, 252, 1000
mcs_train = MCS(df_mspe_train, alpha, w, B)
mcs_train.run()
print(mcs_train.pvalues)
print(mcs_train.included)

1
# for included in mcs.included:
#     df_mcs_included_train.loc[included, key] = '$^{\star}$'
# df_mcs_pvalue_train.loc[:, key] = r"\tiny{[" + mcs.pvalues.loc[model_names].apply(lambda x: truncate(x)).astype(str) + ']}'
#
# mcs = MCS(df_test, alpha, w, B)
# mcs.run()
# for included in mcs.included:
#     df_mcs_included_test.loc[included, key] = '$^{\star}$'
# df_mcs_pvalue_test.loc[:, key] = r"\tiny{[" + mcs.pvalues.loc[model_names].apply(lambda x: truncate(x)).astype(str) + ']}'
#
# df_mcs_included_train.fillna('$^{}$', inplace=True)
# df_mcs_included_test.fillna('$^{}$', inplace=True)
# df_accuracy_train = df_accuracy_train.round(3).astype(str) + df_mcs_included_train
# df_accuracy_test = df_accuracy_test.round(3).astype(str) + df_mcs_included_test


#
#
#
# model = SeasonalSurfaceSVIModel(df_vols_cube)
# model.fit()
# model.compute_stats()
# model.forecast(horizons=horizons)
# mspe = (model.rmspe.copy() /100) **2
# df_mspe = pd.DataFrame(mspe, index=model._time_line, columns=['Avg.'])
#
# model_name = 'ssvi.pkl'
# p = np.array([5, 25, 50, 75, 95]) /100
# dump(model.df_residuals_vols, os.path.join(picklefolder, 'fitted_models', ticker, 'errors', model_name))
# dump(model.df_rel_squared_residuals_vols,os.path.join(picklefolder, 'fitted_models', ticker, 'squared_relative_errors', model_name))
# dump(df_mspe, os.path.join(picklefolder, 'fitted_models', ticker, 'daily_avg_squared_relative_errors', model_name))
# stats = pd.concat([pd.DataFrame(model.rmse), pd.DataFrame(model.rmspe), pd.DataFrame(model.mae),
#                            pd.DataFrame(model.mape), pd.DataFrame(model.r2)], axis=1)
# stats.columns = ['RMSE', 'RMSPE', 'MAE', 'MAPE', 'R2']
# stats_summary = pd.concat([stats.iloc[10:in_sample].describe(percentiles=p).T,
#                            stats.iloc[in_sample:].describe(percentiles=p).T], axis=0)
# stats_summary = stats_summary.drop(columns=['count'])
# dump(stats_summary, os.path.join(picklefolder, 'fitted_models', ticker, 'statistics', model_name))
#
# forecast_objects = [model._dict_rmspe, model._dict_rel_squared_residuals_vols]
# # dump(forecast_objects, os.path.join(picklefolder, 'fitted_models', ticker, 'forecast_errors', model_name))
#
# dict_models = dict()
# for variance in variances:
#     print(variance)
#     mean = 'carr_wu_disprop2_s4'
#     model_name = f'{mean}_{variance}.pkl'
#     dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
#     model = DynamicSurfaceModels(df_surface=df_vols_cube)
#     model.specify_measurement_equation(mean=mean, variance=variance,
#                                        moneyness_pillars=x_knots, maturity_pillars=tau_knots,
#                                        cyclical_knots=cyclical_knots)
#     model._set_model_params(*dict_results['set_params'])
#
#     try:
#         model.run_filter()
#     except:
#         model.run_filter()
#
#     model._compute_standard_errors(N_fit=in_sample, burn_in=10)
#     print('DONE SE')
#     dict_models[variance] = model
#
#
# titles = [r'$\kappa_{1,t}$', '$\omega_{1,t}$', r'$\nu_t$', r'$\rho_t$',
#           r'$\tilde{\eta}_{1,t}$', r'$\tilde{\eta}_{2,t}$',
#           r'$\delta_{1,t}$', r'$\delta^\ast_{1,t}$', r'$\delta_{2,t}$', r'$\delta^\ast_{2,t}$',
#           r'$\delta_{3,t}$', r'$\delta^\ast_{3,t}$', r'$\delta_{4,t}$', r'$\delta^\ast_{4,t}$',
#           r'$v^2(\tau)$ $(\times 1000)$', r'$u^2(x)$']
# THOUSAND = 1000
# fig, ax = plt.subplots(figsize=(20*1.75, 10*2), ncols=4, nrows=4)
# axs = ax.flatten()
# j = 0
# for ax_ in axs:
#     ax_.axhline(y=0, color=colors[5], label='_nolegend_')
#     j += 1
#
# for i in range(14):
#     axs[i].axvline(x=pd.to_datetime(train_end), color=colors[5], label="_nolegend_", linestyle='dotted')
#     if i == 0:
#         axs[i].text(x=pd.to_datetime('20220920'), y=-5, s=r'In-sample', size=18)
#         axs[i].text(x=pd.to_datetime('20230901'), y=-5, s=r'Out-of-sample', size=18)
#     axs[i].set_title(titles[i], size=30)
#
# ax1 = axs[-2]
# divider = make_axes_locatable(ax1)
# ax2 = divider.new_vertical(size="100%", pad=0.1)
# fig.add_axes(ax2)
# ax2.set_title(titles[14], size=30)
# axs[-1].set_title(titles[15], size=30)
# ax1.set_ylim(0, 1.065)
# ax1.spines['top'].set_visible(False)
# ax2.set_ylim(1.1, 12)
# ax2.tick_params(bottom=False, labelbottom=False)
# ax2.spines['bottom'].set_visible(False)
#
# i = 0
# for variance in dict_models.keys():
#     model = dict_models[variance]
#     M = 500
#     f_sim = np.zeros((model._N, model._k, M))
#     for j in range(M):
#         for t in range(model._N):
#             f_j = np.random.multivariate_normal(model.a[t].flatten(), model.P[t])
#             f_j[1:4] = np.array(map_moments(*f_j[1:4]))[:3]
#             f_sim[t, :, j] = f_j
#
#     lb = np.quantile(f_sim, 0.025, axis=2)
#     ub = np.quantile(f_sim, 1-0.025, axis=2)
#     mean = np.mean(f_sim, axis=2)
#     for j in range(model._k):
#         axs[j].plot(timeline, mean[:, j], color=colors[i], linewidth=1)
#         axs[j].fill_between(timeline, lb[:, j], ub[:, j], alpha=0.5, color=colors[i], label='_nolegend_')
#         axs[j].set_xlim(timeline.min(), timeline.max())
#         axs[j].tick_params(axis='x', labelrotation=30)
#
#
#     if variance == 'iid':
#         ones = np.ones(len(tau_grid))
#         sigma2 = model.H[0,0]
#         lb = sigma2 - 2 * model._standard_errors[-1]
#         ub = sigma2 + 2 * model._standard_errors[-1]
#         ax1.plot(ones*sigma2 * THOUSAND, color=colors[i], linewidth=1)
#         ax1.fill_between(tau_grid.flatten(), lb*ones* THOUSAND, ub*ones* THOUSAND, alpha=0.5, color=colors[i], label='_nolegend_')
#
#         axs[-1].axhline(y=1, color=colors[i], linewidth=1)
#         ax1.set_xlabel(r'$\tau$')
#         axs[-1].set_xlabel(r'$x$')
#
#     else:
#         sigma2_x = model._u_x(x_grid).flatten()
#         sigma2_tau = model._v_tau(tau_grid).flatten()
#         len_x = len(x_knots) - 1
#         len_tau = len(tau_knots)
#         len_h = len_x + len_tau
#         se = model._standard_errors[-len_h:].flatten()
#
#         idx_x = x_knots.flatten() != 0
#         se_x = np.zeros(len_x + 1)
#         se_x[idx_x] = se[:len_x]
#         se_tau = se[len_x:]
#
#         se_x_interp = interp1d(x_knots.flatten(), se_x.flatten(), fill_value='extrapolate')
#         se_tau_interp = interp1d(tau_knots.flatten(), se_tau.flatten(), fill_value='extrapolate')
#
#         lb_sigma_x = sigma2_x - 2 * se_x_interp(x_grid.flatten())
#         ub_sigma_x = sigma2_x + 2 * se_x_interp(x_grid.flatten())
#
#         lb_sigma_tau = sigma2_tau - 2 * se_tau_interp(tau_grid.flatten())
#         ub_sigma_tau = sigma2_tau + 2 * se_tau_interp(tau_grid.flatten())
#
#         for ax_ in [ax1, ax2]:
#             ax_.plot(tau_grid, model._v_tau(tau_grid) * THOUSAND, color=colors[i], linewidth=1)
#             ax_.fill_between(tau_grid, lb_sigma_tau * THOUSAND, ub_sigma_tau * THOUSAND, color=colors[i], alpha=0.5)
#             ax_.scatter(model._v_tau.x, model._v_tau.y * THOUSAND, color=colors[i])
#             ax_.set_xlim(0, tau_grid.max())
#
#         axs[-1].plot(x_grid, model._u_x(x_grid), color=colors[i], linewidth=1)
#         axs[-1].scatter(model._u_x.x, model._u_x.y, color=colors[i])
#         axs[-1].fill_between(x_grid, lb_sigma_x, ub_sigma_x, color=colors[i], alpha=0.5)
#
#         axs[-1].set_xlim(x_grid.min(), x_grid.max())
#
#     i += 1
#
# # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
# d = .005  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
# ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
# kwargs.update(transform=axs[-2].transAxes)  # switch to the bottom axes
# axs[-2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# axs[-2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
# axs[0].legend(['iid', 'MXVN'], loc='lower left', prop={'size': 20})
# plt.tight_layout()
# # plt.show()
# plt.savefig(os.path.join(plotfolder, 'model_estimates.pdf'))


titles = [r'$\kappa_{1,t}$', '$\omega_{1,t}$', r'$\nu_t$', r'$\rho_t$',
          r'$\tilde{\eta}_{1,t}$', r'$\tilde{\eta}_{2,t}$',
          r'$\delta_{1,t}$', r'$\delta^\ast_{1,t}$', r'$\delta_{2,t}$', r'$\delta^\ast_{2,t}$',
          r'$\delta_{3,t}$', r'$\delta^\ast_{3,t}$', r'$\delta_{4,t}$', r'$\delta^\ast_{4,t}$'
]

dict_f = dict()
for model_type in ['nls', 'iid', 'mvn-spline']:
    mean = 'carr_wu_disprop2_s4'
    print(model_type)
    if model_type != 'nls':
        mean = 'carr_wu_disprop2_s4'
        model_name = f'{mean}_{model_type}.pkl'
        dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
        model = DynamicSurfaceModels(df_surface=df_vols_cube)
        model.specify_measurement_equation(mean=mean, variance=model_type,
                                           moneyness_pillars=x_knots, maturity_pillars=tau_knots,
                                           cyclical_knots=cyclical_knots)
        model._set_model_params(*dict_results['set_params'])

        try:
            model.run_filter()
        except:
            model.run_filter()
        f = model.a_t[:, :, 0]

    else:
        model = DynamicSurfaceModels(df_surface=df_vols_cube)
        model.specify_measurement_equation(mean=mean, variance='iid',
                                           moneyness_pillars=x_knots, maturity_pillars=tau_knots,
                                           cyclical_knots=cyclical_knots)

        model.fit(cross_sectional=True, refit_cs=True)

        f = model.f_cs

    f_tmp = f.copy()
    for t in range(model._N):
        f_t = f[t].flatten()
        f_t[1:4] = np.array(map_moments(*f_t[1:4]))[:3].flatten()
        f_tmp[t, :] = f_t.flatten()


    dict_f[model_type] = f_tmp


fig, ax = plt.subplots(figsize=(20*1.75, 10*2), ncols=4, nrows=4)
axs = ax.flatten()
j = 0
for ax_ in axs[:-2]:
    ax_.axhline(y=0, color=colors[5], label='_nolegend_')
    j += 1

for i in range(14):
    axs[i].axvline(x=pd.to_datetime(train_end), color=colors[5], label="_nolegend_", linestyle='dotted')
    if i == 0:
        axs[i].text(x=pd.to_datetime('20220920'), y=-5, s=r'In-sample', size=18)
        axs[i].text(x=pd.to_datetime('20230901'), y=-5, s=r'Out-of-sample', size=18)
    axs[i].set_title(titles[i], size=30)

clrs = ['black', #colors[0],
        colors[0]]
i = 0
for key in ['nls', 'iid']:#dict_f.keys():
    f = dict_f[key]
    # color = 'black' if key == 'nls' else colors[i]
    linewidth = 1 if i ==0 else 1
    for j in range(14):
        axs[j].plot(timeline, f[:, j], color=clrs[i],
                    linewidth=linewidth)
        axs[j].set_xlim(timeline.min(), timeline.max())
        axs[j].tick_params(axis='x', labelrotation=30)
    i +=1

axs[0].legend(['NLRS', 'iid'], loc='lower right', ncol=3, prop={'size': 20})
axs[-1].axis("off")
axs[-2].axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plotfolder, f'filtering_over_nls.pdf'))

# plt.show()
#
#
# dict_models = dict()
# for variance in variances:
#     print(variance)
#     mean = 'carr_wu_disprop2_s4'
#     model_name = f'{mean}_{variance}.pkl'
#     dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
#     model = DynamicSurfaceModels(df_surface=df_vols_cube)
#     model.specify_measurement_equation(mean=mean, variance=variance,
#                                        moneyness_pillars=x_knots, maturity_pillars=tau_knots,
#                                        cyclical_knots=cyclical_knots)
#     model._set_model_params(*dict_results['set_params'])
#
#     try:
#         model.run_filter()
#     except:
#         model.run_filter()
#
#     model.compute_stats()
#
#     dict_models[variance] = model
#
# labels = ['iid', 'MXVN']
#
# fig, ax = plt.subplots(figsize=(15*1.25,5*1.25), ncols=3, subplot_kw=dict(projection="3d"))
# axs = ax.flatten()
# for ax_ in axs:
#     ax_.view_init(azim=30, elev=25)
# locs = [368, 246, 816]
# i = 0
# for key in dict_models.keys():
#     model = dict_models[key]
#     j = 0
#     for loc in locs:
#         t = model._time_line[loc]
#         fit_ws = model.df_fitted_vols.loc[t].values[:, 1:] * HUNDRED
#         fit_wos = model.df_fitted_deseasonalized_vols.loc[t].values[:, 1:] * HUNDRED
#         Y = model.df_vols.loc[t].values[:, 1:] * HUNDRED
#         if i == 0:
#             ax[j].scatter(model.vec(model.X), model.vec(model.TAU), model.vec(Y), color=colors[2],
#                              label=r'$I_t$')
#         ax[j].plot_surface(model.X, model.TAU, fit_ws,  color=colors[i], alpha=0.5,
#                                 label=r'$\widehat{I}_{t|t}$ ' + labels[i])
#         ax[j].set_xlim(x_grid.max(), x_grid.min())
#         ax[j].set_xlabel(r'$x$', labelpad=10)
#         ax[j].set_ylabel(r'$\tau$', labelpad=10)
#         ax[j].zaxis.set_rotate_label(False)
#         ax[j].set_zlabel(r'$I_t(x, \tau)$ (%)', rotation=90)
#         ax[j].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
#         j += 1
#     i += 1
# ax[0].legend(ncol=3, prop={'size': 15}, loc='upper right')
# plt.tight_layout(pad=2)
# plt.savefig(os.path.join(plotfolder, 'surface_fits_a.pdf'))
#
# x_to_plot = [0.]
# x_labels_o = [ r'$I_t(0, \tau)$', ]
# x_labels_f = [r'$\widehat{I}_{t|t}(0, \tau)$ ']
# fig, ax = plt.subplots(figsize=(15*1.25,5), ncols=3)
# i = 0
# for key in dict_models.keys():
#     model = dict_models[key]
#     j = 0
#     for loc in locs:
#         t = model._time_line[loc]
#         fit_ws = model.df_fitted_vols.loc[t].values[:, 1:] * HUNDRED
#         fit_wos = model.df_fitted_deseasonalized_vols.loc[t].values[:, 1:] * HUNDRED
#         Y = model.df_vols.loc[t].values[:, 1:] * HUNDRED
#
#         c = 0
#         for k in range(model._m):
#             x_j = model._x_grid[k]
#             if x_j in x_to_plot:
#                 if i ==0:
#                     ax[j].scatter(model._tau_grid.flatten(), Y[k], color=colors[2], label=x_labels_o[c])
#                 ax[j].plot(model._tau_grid.flatten(), fit_ws[k], linestyle='dashed', linewidth=2,
#                               color=colors[i],label=x_labels_f[c] + labels[i])
#                 c+=1
#         ax[j].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
#         ax[j].set_ylabel(r'$I_t(x, \tau)$ (%)', rotation=90)
#         ax[j].set_xlabel(r'$\tau$')
#         j += 1
#     i += 1
# ax[0].legend(loc='upper right', prop={'size': 15})
# plt.tight_layout()
# plt.savefig(os.path.join(plotfolder, 'surface_fits_b.pdf'))
#
#
# fig, ax = plt.subplots(figsize=(15*1.25,10), ncols=3, nrows=2, width_ratios=[1, 1, 1.05], sharey='col', sharex=True)
# i = 0
# for key in dict_models.keys():
#     model = dict_models[key]
#     j = 0
#     for loc in locs:
#         t = model._time_line[loc]
#         fit_ws = model.df_fitted_vols.loc[t].values[:, 1:] * HUNDRED
#         fit_wos = model.df_fitted_deseasonalized_vols.loc[t].values[:, 1:] * HUNDRED
#         Y = model.df_vols.loc[t].values[:, 1:] * HUNDRED
#
#         cf = ax[i, j].contourf(model.X, fit_wos, model.TAU, cmap=cm,
#                                levels=np.linspace(tau_grid.min(), tau_grid.max(), 50), alpha=.5)
#         if j == 2:
#             cax = make_axes_locatable(ax[i, j]).append_axes("right", size="5%", pad="2%")
#             cb = fig.colorbar(cf, cax=cax)
#             cb.set_label(r'$\tau$', rotation=0, labelpad=10)
#
#         ax[i, j].set_ylabel(r'$\widehat{I}^{R}_{t|t}(x, \tau)$ (%)', rotation=90)
#         if i == 1:
#             ax[i, j].set_xlabel(r'$x$')
#
#         ax[i,j].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}: {labels[i]}', size=25)
#
#         j += 1
#
#     i += 1
#
# plt.tight_layout()
# plt.savefig(os.path.join(plotfolder, 'surface_fits_c.pdf'))

# x_to_plot = [0]
# x_labels_o = [ r'$I_t(0, \tau)$', ]
# x_labels_f = [r'$\widehat{I}_{t|t}(0, \tau)$']
# locs = [368, 246, 816]
# fig, ax = plt.subplots(figsize=(5,15), ncols=3, nrows=3, sharex='col', width_ratios=[1,1.25,1.25])
# idx_3d = [1, 4, 7]
#
# i = 0
# for loc in locs:
#     print(i)
#     ax[i, 0].axis('off')
#     ax[i, 0] = plt.subplot(int(f'33{idx_3d[i]}'), projection='3d')
#
#     t = model._time_line[loc]
#     fit_ws = model.df_fitted_vols.loc[t].values[:, 1:] *HUNDRED
#     fit_wos = model.df_fitted_deseasonalized_vols.loc[t].values[:, 1:] * HUNDRED
#     Y = model.df_vols.loc[t].values[:, 1:] * HUNDRED
#
#     c = 0
#     for j in range(model._m):
#         x_j = model._x_grid[j]
#         if x_j in x_to_plot:
#             ax[i, 1].scatter(model._tau_grid.flatten(), Y[j], color=colors[c], label=x_labels_o[c])
#             ax[i, 1].plot(model._tau_grid.flatten(), fit_ws[j], linestyle='dashed', linewidth=2,
#                           color=colors[c],label=x_labels_f[c],)
#             c+=1
#
#     cax = make_axes_locatable(ax[i, 2]).append_axes("right", size="5%", pad="2%")
#     cf = ax[i, 2].contourf(model.X, fit_wos, model.TAU, cmap=cm, levels=np.linspace(tau_grid.min(), tau_grid.max(), 50), alpha=.5)
#     cb = fig.colorbar(cf, cax=cax)
#     cb.set_label(r'$\tau$', rotation=0, labelpad=10)
#
#     ax[i,0].view_init(azim=45, elev=25)
#     ax[i, 0].scatter(model.vec(model.X), model.vec(model.TAU), model.vec(Y), c=cm(model.vec(Y)/HUNDRED), label=r'$I_t$')
#     ax[i, 0].plot_wireframe(model.X, model.TAU, fit_ws, rstride=2, cstride=2, color="k", lw=0.25, label=r'$\widehat{I}_{t|t}$')
#     ax[i, 0].set_xlim(x_grid.max(), x_grid.min())
#
#     ax[i, 0].set_xlabel(r'$x$', labelpad=10)
#     ax[i, 0].set_ylabel(r'$\tau$', labelpad=10)
#     ax[i, 0].zaxis.set_rotate_label(False)
#     ax[i, 0].set_zlabel(r'$I_t(x, \tau)$ (%)', rotation=90)
#
#     ax[i, 1].set_ylabel(r'$I_t(x, \tau)$ (%)', rotation=90)
#     ax[i, 2].set_ylabel(r'$\widehat{I}^{DS}_{t|t}(x, \tau)$ (%)', rotation=90)
#     ax[i, 1].set_xlabel(r'$\tau$')
#     ax[i, 2].set_xlabel(r'$x$')
#
#     ax[i, 0].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
#     ax[i, 1].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
#     ax[i, 2].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
#
#     # ax[0].set_xlabel('Maturity (Years)')
#     # ax[1].set_xlabel('Log-Moneyness')
#     # ax[0].legend(ncol=3, prop={'size': 15})
#
#
#     i +=1
#     # ax[1].legend(ncol=3, prop={'size': 15})
# ax[0,0].legend(loc='upper right', prop={'size': 15})
# ax[0,1].legend(loc='upper right', prop={'size': 15},ncol=2)
# plt.tight_layout(pad=1.5)
# # plt.show()
# plt.savefig(os.path.join(plotfolder, 'surface_fits.pdf'))


#
# # #
# # #
# # #
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # m, n = model._m, model._n
# # X = model._x_grid * np.ones((m,n))
# # Tau = model._tau_grid.T * np.ones((m,n))
# #
# # z = (model.df_rel_squared_residuals_vols.groupby(by='CLOSEST_MONEYNESS').mean()**.5 *100).values
# # z_min = 0
# # z_max = np.ceil(z.max())
# #
# # fig, ax = plt.subplots(figsize=(12,10))
# # c = ax.pcolormesh(Tau, X, z, vmin=z_min, vmax=z_max, shading='gouraud')
# # ax.set_title('pcolormesh')
# # ax.axis([Tau.min(), Tau.max(), X.min(), X.max()])
# # fig.colorbar(c, ax=ax,boundaries=np.linspace(0,z_max,33))
# # plt.tight_layout()
# # plt.show()
# #
# #
#
# #
# #
# # rmspe_x.reset_index().pivot_table(index='DATE', columns=['CLOSEST_MONEYNESS'], values=0)[[-0.4, 0., 0.4]].rolling(5).mean().plot(figsize=(20,10)), plt.tight_layout(), plt.show()
# #
# # fig, ax = plt.subplots(figsize=(20,10))
# # rmpspe_tau.iloc[:, rmpspe_tau.columns <=3/12].mean(axis=1).rolling(5).mean().plot(ax=ax)
# # rmpspe_tau.iloc[:, (rmpspe_tau.columns >3/12) & (rmpspe_tau.columns <=1/2)].mean(axis=1).rolling(5).mean().plot(ax=ax)
# # rmpspe_tau.iloc[:, (rmpspe_tau.columns >6/12) & (rmpspe_tau.columns <=1)].mean(axis=1).rolling(5).mean().plot(ax=ax)
# # rmpspe_tau.iloc[:, rmpspe_tau.columns >1].mean(axis=1).rolling(5).mean().plot(ax=ax)
# # plt.tight_layout(), plt.show()
# #
# # fig, ax = plt.subplots(figsize=(20,10))
# # loc1 = rmpspe_tau.columns <=3/12
# # loc2 = (rmpspe_tau.columns >3/12) & (rmpspe_tau.columns <=1/2)
# # loc3 = (rmpspe_tau.columns >6/12)
# # rmpspe_tau.groupby(by=rmpspe_tau.index.month).mean().iloc[:, loc1].mean(axis=1).plot(ax=ax)
# # rmpspe_tau.groupby(by=rmpspe_tau.index.month).mean().iloc[:, loc2].mean(axis=1).plot(ax=ax)
# # rmpspe_tau.groupby(by=rmpspe_tau.index.month).mean().iloc[:, loc3].mean(axis=1).plot(ax=ax)
# # plt.tight_layout(), plt.show()
# #
#
# #
# #
# #
# # fig, ax = plt.subplots(figsize=(20,10), ncols=2)
# # for mean in [ 'carr_wu_disprop_s1', 'carr_wu_disprop_s2', 'carr_wu_disprop_s3',  'carr_wu_disprop_s4',
# #               'carr_wu_disprop2_s3', 'carr_wu_disprop2_s4',
# #               ]:
# #     variance = 'mvn-spline'
# #     model_name = f'{mean}_{variance}.pkl'
# #     dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, model_name))
# #     model = DynamicSurfaceModels(df_surface=df_vols_cube)
# #     model.specify_measurement_equation(mean=mean, variance=variance,
# #                                        moneyness_pillars=x_knots, maturity_pillars=tau_knots,
# #                                        cyclical_knots=cyclical_knots)
# #     model._set_model_params(*dict_results['set_params'])
# #
# #     ax[0].plot(tau_grid, model._v_tau(tau_grid))
# #     ax[1].plot(x_grid, model._u_x(x_grid))
# #     ax[0].scatter(model._v_tau.x, model._v_tau.y)
# #     ax[1].scatter(model._u_x.x, model._u_x.y)
# #     plt.suptitle(f'{mean}_{variance}', size=25)
# #
# # plt.tight_layout()
# # plt.show()
#
#
# mean = 'carr_wu_disprop2_s4'
# variance = 'iid'
# model_name = f'{mean}_{variance}.pkl'
# dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
# model = DynamicSurfaceModels(df_surface=df_vols_cube)
# model.specify_measurement_equation(mean=mean, variance=variance,
#                                    moneyness_pillars=x_knots, maturity_pillars=tau_knots,
#                                    cyclical_knots=cyclical_knots)
# model._set_model_params(*dict_results['set_params'])
#
#
# try:
#     model.run_filter()
# except:
#     model.run_filter()
#
# model.compute_stats()
# # from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# # for t in [
# #     '2021-03-11', '2021-06-11',
# #           '2021-10-05', '2021-10-06', '2021-10-07', '2021-10-08',
# #           '2021-12-13', '2022-06-13',
# #           '2022-08-11'
# #           ]:
# #     fit_ws = model.df_fitted_vols.loc[t].values[:, 1:]
# #     fit_wos = model.df_fitted_deseasonalized_vols.loc[t].values[:, 1:]
# #     Y = model.df_vols.loc[t].values[:, 1:]
# #
# #     fig, ax = plt.subplots(figsize=(20, 10), ncols=2, sharey=True)
# #     cax = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad="2%")
# #     i = 0
# #     for i in range(model._m):
# #         ax[0].scatter(model._tau_grid.flatten(), Y[i], color=cm(np.linspace(0, 1, model._m)[i]))  # , label=list_T[i])
# #         ax[0].plot(model._tau_grid.flatten(), fit_ws[i], linestyle='dashed', color=cm(np.linspace(0, 1, model._m)[i]), label=f'x={x_grid[i]}')
# #         i += 1
# #
# #     cf = ax[1].contourf(model.X, fit_wos, model.TAU, cmap=cm, levels=np.linspace(tau_grid.min(), tau_grid.max(), 50), alpha=.5)
# #
# #     ax[0].set_xlabel('Maturity (Years)')
# #     ax[1].set_xlabel('Log-Moneyness')
# #     ax[0].legend(ncol=3, prop={'size': 15})
# #
# #     # draw new colorbar in existing cax
# #     cb = fig.colorbar(cf, cax=cax)
# #
# #
# #     # ax[1].legend(ncol=3, prop={'size': 15})
# #     plt.tight_layout()
# #     plt.show()
# #
# #
# #
# # fit_ws = model.df_fitted_vols.groupby(by='CLOSEST_MONEYNESS').mean().values
# # fit_wos = model.df_fitted_deseasonalized_vols.groupby(by='CLOSEST_MONEYNESS').mean().values
# # Y = model.df_vols.groupby(by='CLOSEST_MONEYNESS').mean().values
#
# # fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
# # i = 0
# # for i in range(model._m):
# #     ax[0].scatter(model._tau_grid.flatten(), Y[i], color=cm(np.linspace(0, 1, model._m)[i]))  # , label=list_T[i])
# #     # ax[0].plot(model._tau_grid.flatten(), fit_ws[i], linestyle='dashed', color=cm(np.linspace(0, 1, model._m)[i]), label=f'x={x_grid[i]}')
# #     i += 1
# #
# # i = 0
# # for tau in tau_grid:
# #     ax[1].plot(model._x_grid, fit_wos[:, i], linestyle='dashed', color=cm(tau), label=f'tau={tau}')
# #     i += 1
# #
# # ax[0].set_xlabel('Maturity (Years)')
# # ax[1].set_xlabel('Log-Moneyness')
# # ax[0].legend(ncol=3, prop={'size': 15})
# # # ax[1].legend(ncol=3, prop={'size': 15})
# # plt.tight_layout()
# # plt.show()
# #
# #
# #
# #
# #
# # x_to_plot = [ -0.3,0,0.3,]
# # x_labels_o = [r'$I_t(-0.3, \tau)$', r'$I_t(0, \tau)$', r'$I_t(0.3, \tau)$']
# # x_labels_f = [r'$\widehat{I}_{t|t}(-0.3, \tau)$', r'$\widehat{I}_{t|t}(0, \tau)$', r'$\widehat{I}_{t|t}(0.3, \tau)$']
# # locs = [368, 246, 816]
# # fig, ax = plt.subplots(figsize=(20,15), ncols=3, nrows=3, sharex='col', width_ratios=[1,1.25,1.25])
# # idx_3d = [1, 4, 7]
# #
# # i = 0
# # for loc in locs:
# #     print(i)
# #     ax[i, 0].axis('off')
# #     ax[i, 0] = plt.subplot(int(f'33{idx_3d[i]}'), projection='3d')
# #
# #     t = model._time_line[loc]
# #     fit_ws = model.df_fitted_vols.loc[t].values[:, 1:] *HUNDRED
# #     fit_wos = model.df_fitted_deseasonalized_vols.loc[t].values[:, 1:] * HUNDRED
# #     Y = model.df_vols.loc[t].values[:, 1:] * HUNDRED
# #
# #     c = 0
# #     for j in range(model._m):
# #         x_j = model._x_grid[j]
# #         if x_j in x_to_plot:
# #             ax[i, 1].scatter(model._tau_grid.flatten(), Y[j], color=colors[c], label=x_labels_o[c])
# #             ax[i, 1].plot(model._tau_grid.flatten(), fit_ws[j], linestyle='dashed', linewidth=2,
# #                           color=colors[c],label=x_labels_f[c],)
# #             c+=1
# #
# #     cax = make_axes_locatable(ax[i, 2]).append_axes("right", size="5%", pad="2%")
# #     cf = ax[i, 2].contourf(model.X, fit_wos, model.TAU, cmap=cm, levels=np.linspace(tau_grid.min(), tau_grid.max(), 50), alpha=.5)
# #     cb = fig.colorbar(cf, cax=cax)
# #     cb.set_label(r'$\tau$', rotation=0, labelpad=10)
# #
# #     ax[i,0].view_init(azim=45, elev=25)
# #     ax[i, 0].scatter(model.vec(model.X), model.vec(model.TAU), model.vec(Y), c=cm(model.vec(Y)/HUNDRED), label=r'$I_t$')
# #     ax[i, 0].plot_wireframe(model.X, model.TAU, fit_ws, rstride=2, cstride=2, color="k", lw=0.25, label=r'$\widehat{I}_{t|t}$')
# #     ax[i, 0].set_xlim(x_grid.max(), x_grid.min())
# #
# #     ax[i, 0].set_xlabel(r'$x$', labelpad=10)
# #     ax[i, 0].set_ylabel(r'$\tau$', labelpad=10)
# #     ax[i, 0].zaxis.set_rotate_label(False)
# #     ax[i, 0].set_zlabel(r'$I_t(x, \tau)$ (%)', rotation=90)
# #
# #     ax[i, 1].set_ylabel(r'$I_t(x, \tau)$ (%)', rotation=90)
# #     ax[i, 2].set_ylabel(r'$\widehat{I}^{DS}_{t|t}(x, \tau)$ (%)', rotation=90)
# #     ax[i, 1].set_xlabel(r'$\tau$')
# #     ax[i, 2].set_xlabel(r'$x$')
# #
# #     ax[i, 0].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
# #     ax[i, 1].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
# #     ax[i, 2].set_title(r'$t=$' + f'{t.year}-{t.month}-{t.day}', size=25)
# #
# #     # ax[0].set_xlabel('Maturity (Years)')
# #     # ax[1].set_xlabel('Log-Moneyness')
# #     # ax[0].legend(ncol=3, prop={'size': 15})
# #
# #
# #     i +=1
# #     # ax[1].legend(ncol=3, prop={'size': 15})
# # ax[0,0].legend(loc='upper right', prop={'size': 15})
# # ax[0,1].legend(loc='upper right', prop={'size': 15},ncol=2)
# # plt.tight_layout(pad=1.5)
# # # plt.show()
# # plt.savefig(os.path.join(plotfolder, 'surface_fits.pdf'))
# #
# #
# # labels = ['2W', '1M', '3M', '6M', '1Y', '18M', '2Y']
# # tau_to_plot = [0.0417, 0.0833, 0.25, 0.5, 1, 1.5, 2]
# # df_atm = model.df_vols[model.df_vols.CLOSEST_MONEYNESS == 0.0].iloc[:,1 :]
# # df_atm_ws = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0.0].iloc[:,1 :]
# # df_atm_wos = model.df_fitted_deseasonalized_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == 0.0].iloc[:,1 :]
# #
# # fig, ax = plt.subplots(figsize=(20,15), nrows=3, sharex=True)
# # for tau in tau_grid:
# #     ax[0].scatter(model._time_line, df_atm[tau].values * HUNDRED, s=10, alpha=0.1, color=cm(tau / 2.2), label='_nolegend_')
# # i = 0
# # for tau in tau_to_plot:
# #     ax[0].plot(model._time_line, df_atm_ws[tau].values * HUNDRED, color=cm(tau/2.2),  label=labels[i])
# #     ax[1].plot(model._time_line, df_atm_wos[tau].values * HUNDRED,  color=cm(tau/2.2), label='_nolegend_')
# #     ax[2].plot(model._time_line, (df_atm_ws[tau].values-df_atm_wos[tau].values) * HUNDRED,  color=cm(tau/2.2))
# #     ax[0].set_xlim(model._time_line.min(), model._time_line.max())
# #     ax[1].set_xlim(model._time_line.min(), model._time_line.max())
# #     ax[2].set_xlim(model._time_line.min(), model._time_line.max())
# #     i +=1
# # ax[1].plot(model._time_line, np.exp(model.a_t[:, 2])*HUNDRED, color='black', label=r'$\nu_t$', linestyle='dotted')
# # ax[0].legend(ncol=2)
# # ax[1].legend()
# # ax[0].set_title('Fitted ATM Term-Structure', size=30)
# # ax[1].set_title('Regular Component', size=30)
# # ax[2].set_title('Seasonal Component', size=30)
# # plt.tight_layout()
# # # plt.savefig(os.path.join(plotfolder, 'atm_fit.pdf'))
# # plt.show()
# #
# # for tau in tau_to_plot:
# #     x = df_atm_ws[tau].values
# #     y = (df_atm_ws[tau].values-df_atm_wos[tau].values)
# #     print(np.corrcoef(x, y)[0,1]**2 *HUNDRED)
# #
# #
# # df_vix = pd.read_csv(os.path.join(datafolder, 'indices', 'VIX.csv')).dropna()
# # df_vix.DATE = pd.to_datetime(df_vix.DATE)
# # df_vix.set_index('DATE', inplace=True)
# # df_vix = df_vix.reindex(df_atm.index)
# # df_vix = df_vix.ffill().bfill()
# #
# # fig, ax = plt.subplots(figsize=(20,10))
# # ax_twin = ax.twinx()
# # ax_twin.plot(df_vix, color=colors[2])
# # # ax.plot(pd.DataFrame(np.exp(model.a_t[:, 2]), index=timeline))
# # ax.plot(df_atm[[0.0417, 0.0833, 0.125]].mean(axis=1))
# # plt.tight_layout()
# # plt.show()
# #
# # import statsmodels.api as sm
# # for tau in tau_to_plot:
# #     y = df_atm[[0.0417, 0.0833, 0.125]].mean(axis=1).interpolate().diff()
# #     x = df_vix.diff().dropna()
# #     # x = x.clip(*(x.quantile(0.01), x.quantile(0.9)), axis=1)
# #     fig, ax = plt.subplots(figsize=(20,10))
# #     ax_twin = ax.twinx()
# #     ax.plot(y)
# #     ax_twin.plot(x, color=colors[2])
# #     plt.tight_layout()
# #     plt.show()
# #     X = sm.add_constant(x)
# #     ols_model = sm.OLS(y, X)
# #     fit_ols = ols_model.fit()#(cov_type='HAC',cov_kwds={'maxlags':6})
# #     print(fit_ols.summary())
# #     # print(np.corrcoef(x, y)[0,1]**2 *HUNDRED)
# #
# # # for t in range(252 , len(timeline)):
# # #     X_
# # tmp = pd.concat([x, y], axis=1).rolling(100).corr().reset_index()
# # tmp[tmp.level_1 == 'VIX'][[0]].plot(figsize=(20,10)), plt.tight_layout(), plt.show()
# #
# #
# #
# # df_hhng_spot = pd.read_csv(os.path.join(datafolder, 'spots', 'DHHNGSP.csv'))
# # df_hhng_spot = df_hhng_spot.set_index(pd.to_datetime(df_hhng_spot.DATE), drop=True).dropna()[['DHHNGSP']]
# #
# # fcc = ForwardCurveCalculator(os.path.join(datafolder, 'comdty_forwards'))
# # zrc = ZeroRatesCalculator(os.path.join(datafolder, 'rates'))
# #
# # df_forwards = fcc.interpolate(tau_grid, timeline)
# # df_rates = zrc.interpolate(tau_grid, timeline)
# #
# # fig, ax = plt.subplots(figsize=(20,10), ncols=3, nrows=3, sharex=True)
# # axs = ax.flatten()
# # i = 0
# # for t in ['2020-08-31', '2020-11-16', '2021-02-05',
# #              '2021-10-06', '2022-04-07', '2022-12-08',
# #              '2023-06-29', '2024-01-04',  '2024-08-15']:
# #
# #     axs[i].scatter(tau_grid, df_atm.loc[t] * HUNDRED)
# #     axs[i].plot(tau_grid, df_atm_ws.loc[t] * HUNDRED)
# #     axs[i].set_title(f't={t}', size=25)
# #     axs_i_twin = axs[i].twinx()
# #     axs_i_twin.plot(df_forwards.columns, df_forwards.loc[t].values, color=colors[2], linestyle='dotted')
# #     axs_i_twin.axhline(y =df_hhng_spot.loc[t].values, color='black', linestyle='dotted')
# #
# #     i +=1
# # plt.tight_layout()
# # plt.show()
# #
# #
# #
# # fig, ax = plt.subplots(figsize=(20*1.25,10*1.25), ncols=3, nrows=2, sharex=True, sharey='row')
# # axs = ax.flatten()
# # i = 0
# # for x in [0.025, 0.05, 0.1, 0.2, 0.4]:
# #     i = 0
# #     Ic_true = model.df_vols[model.df_vols.CLOSEST_MONEYNESS ==x]
# #     Ip_true = model.df_vols[model.df_vols.CLOSEST_MONEYNESS == -x]
# #     Ic_fit = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS ==x]
# #     Ip_fit = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == -x]
# #     for tau in key_ttm:
# #         skew_true = (Ic_true[tau] - Ip_true[tau])/(2*x)
# #         skew_fit = (Ic_fit[tau] - Ip_fit[tau])/(2*x)
# #         skew_rmspe = ((skew_true/skew_fit - 1) **2).median() **.5
# #         print(x, tau, skew_rmspe *100)
# #         axs[i].scatter(timeline, skew_true.values.flatten(), alpha=0.5, s=10)
# #         axs[i].plot(timeline, skew_fit.values.flatten(),)
# #         axs[i].set_xlim(timeline.min(), timeline.max())
# #         i+=1
# #
# # plt.tight_layout()
# # plt.show()
# #
# # fig, ax = plt.subplots(figsize=(20,10))
# # x = 0.025
# # for tau in [0.25]:
# #     Ic_true = model.df_vols[model.df_vols.CLOSEST_MONEYNESS == x]
# #     Ip_true = model.df_vols[model.df_vols.CLOSEST_MONEYNESS == -x]
# #     Ic_fit = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == x]
# #     Ip_fit = model.df_fitted_vols[model.df_fitted_vols.CLOSEST_MONEYNESS == -x]
# #     skew_true = (Ic_true[tau] - Ip_true[tau])
# #     skew_fit = (Ic_fit[tau] - Ip_fit[tau])
# #     ax.scatter(timeline, skew_true.values.flatten(), alpha=0.5, s=10)
# #     ax.plot(timeline, skew_fit.values.flatten(), )
# #     ax.set_xlim(timeline.min(), timeline.max())
# #
# #
# # plt.tight_layout()
# # plt.show()
# #
# #
# # df_rmspe = model.df_rel_squared_residuals_vols.groupby(by='CLOSEST_MONEYNESS').mean() **.5
# #
# # df_rmspe.index = pd.cut(df_rmspe.index, bins=[-0.4, -0.2, -0.05, 0.05, 0.2, 0.4], include_lowest=True)
# # df_rmspe.columns = pd.cut(df_rmspe.columns,  bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
# # df_rmspe = df_rmspe.reset_index().groupby(by='index').mean()
# # df_rmspe = df_rmspe.T.reset_index().groupby(by='index').mean().T *HUNDRED
# # df_rmspe['AVG'] = df_rmspe.mean(axis=1)
# # df_rmspe.loc['AVG'] = df_rmspe.mean(axis=0)
# # df_rmspe.style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, f'avg_rmspe_{mean}_{variance}.tex'))
# #
# #
# #
# #
# # fig, ax = plt.subplots(figsize=(20*1.25,10*1.25), ncols=2, sharey=True)
# #
# # rmspe_x = model.df_rel_squared_residuals_vols.reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS']).mean(axis=1) ** .5 * 100
# # rmspe_x = rmspe_x.reset_index().pivot_table(index='DATE', columns='CLOSEST_MONEYNESS', values=0)
# # rmspe_x.columns = pd.cut(np.abs(rmspe_x.columns), bins=[0, 0.05, 0.1, 0.2,0.3,  0.4], include_lowest=True)
# # rmspe_x = rmspe_x.T.reset_index().groupby(by='index').mean().T
# # rmspe_x_lb = rmspe_x.iloc[:, 0]
# # rmspe_x_ub = rmspe_x.iloc[:, -1]
# #
# # rmpspe_tau = model.df_rel_squared_residuals_vols.drop(columns=['CLOSEST_MONEYNESS']).groupby(by='DATE').mean() ** .5 * 100
# # rmpspe_tau.columns = pd.cut(rmpspe_tau.columns,  bins=[0, 0.25, 0.75, 1, 2.5])
# # rmpspe_tau = rmpspe_tau.T.reset_index().groupby(by='index').mean().T
# # rmspe_tau_lb = rmpspe_tau.iloc[:, 0]
# # rmspe_tau_ub = rmpspe_tau.iloc[:, -1]
# #
# #
# # for i in range(2):
# #     ax[i].axhline(y=0, color=colors[5], label='_nolegend_')
# #     ax[i].axvline(x=pd.to_datetime(train_end), color='black', linestyle='dotted', label='_nolegend_')
# #     ax[i].plot(model._time_line, model.rmspe, color=colors[2], linewidth=1)
# #     ax[i].axhline(y=model.rmspe.mean(), color='black', label='Avg.', linewidth=1)
# #     ax[i].set_xlim(timeline.min(), timeline.max())
# #
# # ax[0].plot(model._time_line, rmspe_x_lb, color=colors[0], linewidth=1)
# # ax[0].plot(model._time_line, rmspe_x_ub, color=colors[1], linewidth=1)
# # ax[0].fill_between(rmspe_x_lb.index, rmspe_x_lb.values, rmspe_x_ub.values, color=colors[5], alpha=0.25)
# #
# # ax[1].plot(model._time_line, rmspe_tau_lb, color=colors[0], linewidth=1)
# # ax[1].plot(model._time_line, rmspe_tau_ub, color=colors[1], linewidth=1)
# # ax[1].fill_between(rmspe_tau_lb.index, rmspe_tau_lb.values, rmspe_tau_ub.values, color=colors[5], alpha=0.25)
# #
# # ax[0].legend([r'RMSPE$_t$', r'RMSPE$_t$ Avg.', r'RMSPE$_t()$'])
# # ax[0].set_ylim(0, 25)
# # plt.tight_layout()
# # plt.show()
# #
# #
# bins_x = dict({-0.4: '1', -0.3: '1',
#                -0.2: '2', -0.1: '2',
#                 -0.05: '3', -0.025: '3',  0.0:'3',  0.025:'3', 0.05:'3',
#                 0.1:'4',  0.2: '4',
#                0.3:'5', 0.4:'5'})
# df_errors = model.df_squared_residuals_vols.copy()
# df_errors = df_errors[df_errors.index <= train_end]
# # df_errors = df_errors[df_errors.index > train_end]
#
# df_errors[['CLOSEST_MONEYNESS']].replace(bins_x)
# df_errors[['CLOSEST_MONEYNESS']] = df_errors[['CLOSEST_MONEYNESS']].replace(bins_x)
# df_errors.set_index('CLOSEST_MONEYNESS', inplace=True)
# df_errors.columns = pd.cut(df_errors.columns,  bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
#
# df_rmspe = df_errors.groupby(by='CLOSEST_MONEYNESS').mean()
# df_rmspe = df_rmspe.T.groupby(level=0).mean().T
# df_rmspe = df_rmspe**.5 * HUNDRED
# df_rmspe['AVG'] = df_rmspe.mean(axis=1)
# df_rmspe.loc['AVG'] = df_rmspe.mean(axis=0)
# print(df_rmspe.round(3).T)