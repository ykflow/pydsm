import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from config_utils.config_utils_main import import_ts_ivs
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from numba.typed import List
from copy import deepcopy
from joblib import Parallel, delayed


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
min_ttm = 5/days
max_ttm = 1
cm = diverge_map(high=colors[2], low=colors[0])


ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
x_grid = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
x_knots = np.array([-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
tau_grid = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
# tau_grid = np.concatenate((np.array([1 / 48]), tau_grid)).round(4)
tau_knots = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12])
cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])
df_vols_cube = import_ts_ivs(x_grid, tau_grid, ticker)
df_vols_cube.iloc[:, 1:] **= 2


means = [
    # 'carr_wu_standard',
         # 'carr_wu_s1_fixed', 'carr_wu_s2_fixed', 'carr_wu_s3_fixed', 'carr_wu_s4_fixed',
         # 'carr_wu_s1_decay', 'carr_wu_s2_decay', 'carr_wu_s3_decay', 'carr_wu_s4_decay',
         # 'carr_wu_s3_decay_kappa', 'carr_wu_s4_decay_kappa',
         # 'carr_wu_s3_decay_omega', 'carr_wu_s4_decay_omega',
         # 'carr_wu_s3_decay_rho', 'carr_wu_s4_decay_rho',
         # 'carr_wu_s3_decay_nu', 'carr_wu_s4_decay_nu',
         # 'carr_wu_s4_decay_kappa_omega', 'carr_wu_s4_decay_kappa_rho', 'carr_wu_s4_decay_kappa_nu',
         # 'carr_wu_s4_decay_omega_rho', 'carr_wu_s4_decay_omega_nu',
         # 'carr_wu_s4_decay_rho_nu',
         # 'carr_wu_s4_decay_kappa_omega_rho', 'carr_wu_s4_decay_kappa_omega_nu', 'carr_wu_s4_decay_kappa_rho_nu',
         # 'carr_wu_s4_decay_omega_rho_nu',
         # 'carr_wu_s3_decay_kappa_omega_rho_nu', 'carr_wu_s4_decay_kappa_omega_rho_nu',
         # 'carr_wu_disprop_s3', 'carr_wu_disprop_s4',
         'carr_wu_disprop2_s3', 'carr_wu_disprop2_s4',
         # 'carr_wu_disprop2_s3_corr', 'carr_wu_disprop2_s4_corr',
         # 'carr_wu_disprop2_s3_corr_nu', 'carr_wu_disprop2_s4_corr_nu',
         'carr_wu_disprop2_sknots'
         ]

seasons = ['1', '2', '3', '4']
season_types = ['fixed', 'decay']
df_models = pd.DataFrame(means, columns=['Model'])
for idx, row in df_models.iterrows():
    label = row['Model']
    if 'fixed' in label:
        col = 'S_FXD'
    elif 'decay' in label:
        col = 'S_DCY'
    else:
        col = None
    if col is not None:
        for s in seasons:
            if s in label:
                df_models.loc[idx, col] = s

    for moment in ['kappa', 'omega', 'rho', 'nu']:
        if moment in label:
            df_models.loc[idx, f'{moment}_DCY'] = '\checkmark'

model = DynamicSurfaceModels(df_surface=df_vols_cube)

def process(i):
    model_i = deepcopy(model)
    model_i.specify_measurement_equation(mean=means[i], variance='iid',
                                       moneyness_pillars=x_knots, maturity_pillars=tau_knots,
                                       cyclical_knots=cyclical_knots)
    model_i.fit(cross_sectional=True)
    rmse = pd.DataFrame(model_i._rmspe_cs, index=model_i._time_line).describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
    rmse = pd.DataFrame(rmse.iloc[1:]).T
    rmse.index = [means[i]]
    return rmse

results = Parallel(n_jobs=3, verbose=3)(delayed(process)(i) for i in range(len(means)))

df_results = pd.concat(results).drop(columns=['min', 'max']).reset_index()
# df_models = df_models.merge(df_results, left_on='Model', right_on='index').drop(columns=['Model', 'index'])
# df_models.style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, 'results_fitted_models_cross_sectional.tex'))

# model = DynamicSurfaceModels(df_surface=df_vols_cube)
# for mean in means:
#
#     model.specify_measurement_equation(mean='mean', variance='iid',
#                                        moneyness_pillars=x_knots, maturity_pillars=tau_knots,
#                                        cyclical_knots=cyclical_knots)
#
# model.fit(cross_sectional=True)
# rmse = pd.DataFrame(model._rmspe_cs, index=model._time_line)
# rmse.plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
# f_hat = pd.DataFrame(model.f_cs, index=model._time_line)
# betas_hat = pd.DataFrame(model.beta_cs, index=model._time_line)
# print(rmse.describe())
# for t in [
#     '2020-07-29',
#     '2021-03-11', '2021-06-11',
#           '2021-10-05', '2021-10-06', '2021-10-07', '2021-10-08',
#           '2021-12-13', '2022-06-13',
#           '2022-08-11'
#           ]:
#     loc = np.argwhere(model._time_line == t).flatten()[0]
#     betas = betas_hat.iloc[loc].values
#     f_ws = f_hat.iloc[loc].values.reshape(model._k,1)
#     f_wos = f_ws.copy()
#     f_wos[7:] = 0
#     t0 = model.day_of_year(pd.to_datetime(t))
#     tmp = model.df_surface.loc[t].sort_values(by='CLOSEST_MONEYNESS')
#     Y = tmp.drop(columns='CLOSEST_MONEYNESS').copy().values **.5
#     m, n = Y.shape
#     x = tmp.CLOSEST_MONEYNESS.values.reshape(m, 1) * np.ones((m, n))
#     tau = tmp.drop(columns='CLOSEST_MONEYNESS').columns.values.reshape(n, 1).T * np.ones((m, n))
#     T = t0 + tau
#     FIXED = List()
#     FIXED.append(model.vec(x))
#     FIXED.append(model.vec(tau))
#     FIXED.append(model.vec(T))
#     FIXED.append(model._cyclical_spline_builder(model.vec(tau)))
#
#     fit_ws = model._builder._Zf(f_ws, FIXED, betas, model._p, model._k).reshape(n, m).T **.5
#     fit_wos = model._builder._Zf(f_wos, FIXED, betas, model._p, model._k).reshape(n, m).T **.5
#
#     fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
#     i = 0
#     for i in range(m):
#         ax[0].scatter(tau[i], Y[i], color=cm(np.linspace(0, 1, m)[i]))  # , label=list_T[i])
#         ax[0].plot(tau[i], fit_ws[i], linestyle='dashed', color=cm(np.linspace(0, 1, m)[i]), label=f'x={x_grid[i]}')
#         i += 1
#
#     i = 0
#     for tau in tau_grid:
#         ax[1].plot(x[:, i], fit_wos[:, i], linestyle='dashed', color=cm(tau), label=f'tau={tau}')
#         i += 1
#
#     ax[0].set_xlabel('Maturity (Years)')
#     ax[1].set_xlabel('Log-Moneyness')
#     ax[0].legend(ncol=3, prop={'size': 15})
#     ax[1].legend(ncol=3, prop={'size': 15})
#     plt.tight_layout()
#     plt.show()
#

#
#     k_fxd_time = 4
#     k_fxd_space = 0
#     k_tvp_time = 0
#     k_tvp_space = 4
#
#     model = DynamicCurveModels(y*HUNDRED, tau, timeline)
#     model.specify_design(k_fxd_time, k_fxd_space, k_tvp_time, k_tvp_space)
#     model.fit()
#
#
# # import random
# # random.seed(7)
# # idx = np.sort(random.sample(timeline.tolist(), 16))
# # fig, ax = plt.subplots(figsize=(20,10), ncols=4, nrows=4, sharex=True,)
# # axs = ax.flatten()
# # i = 0
# # ns_states = model.a_t[:, :3]
# # for t in idx:
# #     loc = np.array(timeline) == t
# #     tau_t, y_t, v_t = model.tau[loc], model.y[loc, :, 0], model._v_t[loc, :, 0]
# #     loc_not_na = np.invert(np.isnan(y_t))
# #     rmspe = np.sqrt(np.mean((v_t[loc_not_na]/y_t[loc_not_na])**2))*100
# #     axs[i].scatter(tau_t[loc_not_na], y_t[loc_not_na])
# #     axs[i].plot(tau_t[loc_not_na], y_t[loc_not_na] - v_t[loc_not_na], color=colors[2])
# #
# #     ttm = np.linspace(1/12, 1.5, 30)
# #     slope = (1 - np.exp(-model.lmbdas * ttm)) / (model.lmbdas * ttm)
# #     curvature = slope - np.exp(-model.lmbdas * ttm)
# #     fit = ns_states[:, 0] + slope * ns_states[:, 1] + curvature * ns_states[:, 2]
# #
# #     axs[i].plot(ttm, pd.DataFrame(fit, index=timeline).loc[t], color=colors[1], linestyle='dashed')
# #     if i >= 12:
# #         axs[i].set_xlabel('Maturity (Years)')
# #     # twinx = axs[i].twinx()
# #     # twinx.plot(expiries, fitted_season, color=colors[3], linestyle='dotted', linewidth=3)
# #     # tmp.plot(x='CLOSEST_EXPIRY', y=0., ax=axs[i], legend=False)
# #     axs[i].set_title(f'{t.strftime("%Y-%m-%d")} ({rmspe:.3f}%)', size=20)
# #     i+=1
# # plt.tight_layout()
# # # plt.show()
# # k = 4
# # plt.savefig(os.path.join(plotfolder, f'dyn_fixed_HH_k={k}.pdf'))
#
#     ns_states = model.a_t[:, :3]
#     ttm = 1/12
#     slope = (1 - np.exp(-model.lmbdas * ttm)) / (model.lmbdas * ttm)
#     curvature = slope - np.exp(-model.lmbdas * ttm)
#     fit = ns_states[:, 0] + slope* ns_states[:, 1] + curvature*ns_states[:, 2]
#
#     df_1m_vols = pd.concat([df_1m_vols, pd.DataFrame(fit, columns=[ticker], index=timeline)], axis=1)
# #
# #
# #
# # # df = pd.DataFrame()
# # # for i in range(3):
# # #     df = pd.concat([df, pd.DataFrame(list_1m_vols[i].flatten(), columns=[tickers[i]])], axis=1)
# #
#
# df_vix =pd.read_excel(os.path.join(datafolder, 'VIXCLS.xlsx'))
# df_vix.columns = ['DATE', 'VIX']
# df_vix['DATE'] = pd.to_datetime(df_vix['DATE'] )
# df_vix.set_index('DATE', inplace=True)
#
# from statsmodels.multivariate.pca import PCA
# pca_model = PCA(df_1m_vols.dropna(), standardize=True, missing='fill-em', ncomp=3)
# fig, ax = plt.subplots(figsize=(20,10))
# ax1 = ax.twinx()
# ax.plot(pca_model.factors.index, pca_model.factors.iloc[:, 0], color=colors[0])
# ax1.plot(df_vix.index, df_vix.iloc[:, 0],color=colors[2])
# ax1.set_xlim(pca_model.factors.index.min(), pca_model.factors.index.max())
# ax1.legend(['VIX'], loc='upper right')
# ax.legend(['PC1'], loc='upper left')
# plt.tight_layout()
# plt.savefig(os.path.join(plotfolder, 'pc1_vix.pdf'))
#
#
# pca_model.plot_rsquare(), plt.show()
#
# # df_model_results = pd.read_csv('df_model_results.csv')
# #
# # tmp = df_model_results[df_model_results.Ticker == tickers[2]].reset_index()
# # print(tmp[tmp.variable == 'BIC'].min())
#
# cm = diverge_map(high=colors[2], low=colors[0])
# ns_states = model.a_t[:, :3]
# fig, ax = plt.subplots(figsize=(20,10))
#
# for t in df.index:
#     tmp = df[df.index == t][df.columns[df.columns <=1]]
#     nobs = tmp.shape[1]
#     plot = ax.scatter([t] * nobs, tmp.values.flatten()*HUNDRED, c=tmp.columns, s=8, cmap=cm, alpha=0.3, vmin=0, vmax=1.01, label='_nolegend_')
#
# i = 0
# for ttm in [1/12, 0.25, 0.5, 0.75, 1,]:
#     slope = (1 - np.exp(-model.lmbdas * ttm)) / (model.lmbdas * ttm)
#     curvature = slope - np.exp(-model.lmbdas * ttm)
#     fit = ns_states[:, 0] + slope* ns_states[:, 1] + curvature*ns_states[:, 2]
#     ax.plot(model.timeline, fit, color=cm(ttm/1.01), label=['1M',  '3M', '6M', '9M', '1Y', '1.5Y'][i], linewidth=2.5)
#     i+=1
# ax.set_xlim(df.index.min(), df.index.max())
# ax.set_title('HH ATM Deseasonalized IV Term-Structure', size=35)
# ax.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(plotfolder, f'HH_deseasoned_iv.pdf'))
#

# from design_matrices.seasonality import SeasonalityDesignMatrix
# from design_matrices.stochastic_seasonality import StochasticSeasonalityDesignMatrix
#
#
# fig, ax = plt.subplots(figsize=(20,10), ncols=2, sharex=True)
# axs = ax.flatten()
# for ax_ in axs:
#     ax_.axhline(y=0, color=colors[5], label='_nolegend_')
#
# sdm = SeasonalityDesignMatrix(k_fxd_time, model.betas, model._mle.cdmb._t)
# Xbeta = sdm.make_design_matrix()
# pd.DataFrame(Xbeta.flatten(), index=timeline).plot(ax=axs[0])
#
# for tau in [1/12,  0.5, 0.75, ]:
#     ssdm = StochasticSeasonalityDesignMatrix(k_tvp_space, model._mle.cdmb._t+tau)
#     Z = ssdm.make_design_matrix()
#     s = Z @ model.a_t[:, 3:3+k_tvp_space*2]
#     pd.DataFrame(s.flatten(), index=timeline).plot(ax=axs[1])
#
# axs[0].legend([r'$\theta_{1,t}$'])
# axs[1].legend([r'$\theta_{2,t}$(t+1M)', r'$\theta_{2,t}$(t+6M)', r'$\theta_{2,t}$(t+9M)'])
# axs[0].set_title('Deterministic Time Seasonality', size=25)
# axs[1].set_title('Stochastic Space Seasonality', size=25)
# axs[0].set_ylim(-5, 5)
# axs[1].set_ylim(-30, 30)
# plt.tight_layout()
# plt.savefig(os.path.join(plotfolder, f'HH_dyn_time_space_seasons.pdf'))
#
#
#
# from design_matrices.seasonality import SeasonalityDesignMatrix
# from design_matrices.stochastic_seasonality import StochasticSeasonalityDesignMatrix
# fig, ax = plt.subplots(figsize=(20,10), ncols=2, sharex=True)
# axs = ax.flatten()
# for ax_ in axs:
#     ax_.axhline(y=0, color=colors[5], label='_nolegend_')
# sdm = SeasonalityDesignMatrix(k_fxd_time, model.betas[:k_fxd_time*2], model._mle.cdmb._t)
# Xbeta = sdm.make_design_matrix()
# pd.DataFrame(Xbeta.flatten(), index=timeline).plot(ax=axs[0])
#
# for tau in [1/12,  0.5, 0.75, ]:
#     sdm = SeasonalityDesignMatrix(k_fxd_space, model.betas[k_fxd_time * 2:], model._mle.cdmb._t+tau)
#     Xbeta = sdm.make_design_matrix()
#     pd.DataFrame(Xbeta.flatten(), index=timeline).plot(ax=axs[1])
# axs[0].legend([r'$\theta_{1,t}$'])
# axs[1].legend([r'$\theta_{2,t}$(t+1M)', r'$\theta_{2,t}$(t+6M)', r'$\theta_{2,t}$(t+9M)'])
# axs[0].set_title('Deterministic Time Seasonality', size=25)
# axs[1].set_title('Deterministic Space Seasonality', size=25)
# axs[0].set_ylim(-5, 5)
# axs[1].set_ylim(-30, 30)
# plt.tight_layout()
# plt.savefig(os.path.join(plotfolder, f'HH_fixed_time_space_seasons.pdf'))




# def tanh(a, lb, ub):
#     tmp = (np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))
#     return lb + (ub-lb)*(tmp/2 +0.5)
#
# def invtanh(a, lb, ub):
#     tmp = 2*(a - lb)/(ub-lb) - 1
#     return np.log((1+tmp)/(1-tmp))/2
#
#
# fig, ax = plt.subplots(figsize=(20,10))
# x = np.linspace(-0.99, 0.999, 100)
# ax.plot(x, invtanh(x, -1, 1))
# plt.tight_layout()
# plt.show()