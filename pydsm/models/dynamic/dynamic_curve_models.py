import numpy as np
from design_matrices.curve_design_matrix_builder import CurveDesignMatrixBuilder
from estimation.mle_curve_models import MLECurveModels
from scipy.optimize import OptimizeResult


class DynamicCurveModels:
    def __init__(self, y: np.array, tau: np.array, timeline):
        self.y = y
        self.tau = tau
        self.timeline = timeline

    def specify_design(self, k_fixed_time_seasons: int = 0, k_fixed_curve_seasons: int = 0,
                       k_tvp_time_seasons: int = 0, k_tvp_curve_seasons: int = 0):

        self.k_fixed_time_seasons = k_fixed_time_seasons
        self.k_fixed_curve_seasons = k_fixed_curve_seasons
        self.k_tvp_time_seasons = k_tvp_time_seasons
        self.k_tvp_curve_seasons = k_tvp_curve_seasons

        self.add_fixed_time_seasons = True if self.k_fixed_time_seasons > 0 else False
        self.add_fixed_curve_seasons = True if self.k_fixed_curve_seasons > 0 else False
        self.add_tvp_time_seasons = True if self.k_tvp_time_seasons > 0 else False
        self.add_tvp_curve_seasons = True if self.k_tvp_curve_seasons > 0 else False
        self._args_builder = (self.add_fixed_time_seasons, self.add_fixed_curve_seasons, self.add_tvp_time_seasons,
                              self.add_tvp_curve_seasons,
                              self.k_fixed_time_seasons, self.k_fixed_curve_seasons, self.k_tvp_time_seasons,
                              self.k_tvp_curve_seasons)

        self.builder = CurveDesignMatrixBuilder(self.tau, self.timeline)
        self.builder.specify_design_matrix(*self._args_builder)

    def fit(self):
        self._mle = MLECurveModels(self.y, self.tau, self.timeline, self.builder)
        self._mle.estimate()
        self._set_model_params(*self._mle.gather_results())
        self._estimates = self._mle.get_untransformed_params()
        self.run_filter()

    def _set_model_params(self, untransformed_params, lmbdas, betas, yx, Z, Xbeta, c, T, R, Q, H, a1, P1, Wx, Ix):
        self.untransformed_params = untransformed_params
        self.lmbdas = lmbdas
        self.betas = betas
        self.yx = yx
        self.Z = Z
        self.Xbeta = Xbeta
        self.c = c
        self.T = T
        self.R = R
        self.Q = Q
        self.H = H
        self.a1 = a1
        self.P1 = P1
        self.n, self.p, self.m = self.Z.shape
        self.Wx = Wx
        self.Ix = Ix

    def run_filter(self):
        if self._mle is None:
            raise ('Fit model first!!!')
        a, P, a_t, P_t, LL = self._mle._filter(self.yx, self.Z, self.Xbeta, self.c, self.T, self.R, self.Q, self.H,
                                               self.a1, self.P1, self.n, self.m, self.Wx, self.Ix)

        self.a = a
        self.P = P
        self.a_t = a_t
        self.P_t = P_t
        self.LL = LL

        self.ll = LL[self._mle._burn_in:].sum()
        k = self.untransformed_params.shape[0]
        nobs = np.invert(np.isnan(self.y)).sum()
        self.aic = 2 * k - 2 * self.ll
        self.bic = k * np.log(nobs) - 2 * self.ll

        n, p, m = self.Z.shape
        v = np.zeros((n, p, 1)) * np.nan
        v_t = np.zeros((n, p, 1)) * np.nan
        for t in range(n):
            v[t] = self.y[t] - self.Z[t] @ self.a[t] - self.Xbeta[t]
            v_t[t] = self.y[t] - self.Z[t] @ self.a_t[t] - self.Xbeta[t]

        self._v = v
        self._v_t = v_t

#
#
# import os
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
# import matplotlib.pyplot as plt
# from data_utils.build_ts_cubes import build_ts_cube
# from models.curve_models import SeasonalNelsonSiegelCurves
# from datetime import datetime, timedelta
# from models.curve_models import SviCurve
# from statsmodels.multivariate.pca import PCA
# from joblib import dump, load
#
#
#
# #### CONFIG
# codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
# basefolder = codefolder.parent
# datafolder = os.path.join(basefolder, 'data_storage', )
# plotfolder = os.path.join(basefolder, 'plot_folder', )
# HUNDRED = 100
# days = 365
# set_theme()
# min_ttm = 5/days
# max_ttm = 1
# cm = diverge_map(high=colors[2], low=colors[0])
# moneyness_grid = np.linspace(-1, 1, 41, endpoint=True).round(2) # 0.05 step size
# # maturity_grid = np.linspace(0, 1, 25, endpoint=True).round(4) #bi-weekly
# # maturity_grid = np.linspace(0, 1, 49, endpoint=True).round(4) #weekly
# maturity_grid = np.linspace(0, 1.5, 73, endpoint=True).round(4)
#
# filtered_dir = 'C:/Users/PI26UT/Documents/Data/ComdtyVols/Filtered Data'
# tickers = [
#     'Crude Futures_Brent 1st Line_BRENT',
#            'NG Pen Futures 25k ICE Lots_Henry_HH',
#           'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH'
# ]
#
# df_1m_vols = pd.DataFrame()
# for ticker in tickers:
#
#
#     df_vols_cube = load(os.path.join(basefolder, 'main_runners', f'{ticker}.pkl'))
#     timeline = df_vols_cube.index.unique()
#     df_vols = df_vols_cube[['CLOSEST_EXPIRY', 0.]]
#     df = df_vols.reset_index().pivot_table(index=df_vols_cube.index, columns='CLOSEST_EXPIRY', values=0.)
#     n, p = df.shape
#     y = df.values.reshape(n, p, 1)
#     tau = np.ones((n, 1)) * df.columns.values.reshape(1, p)
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