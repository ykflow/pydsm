import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from config_utils.config_utils_main import import_ts_ivs
from numba.typed import List
from joblib import dump, load
from datetime import datetime, timedelta


def get_mvn_from_iid_inits(params, k_states, k_ar1_factor, k_betas, x_knots, tau_knots):
    k_x = len(x_knots) -1
    k_tau = len(tau_knots)
    k_state_params = k_ar1_factor*2 + k_states  #c, phi, q
    k = k_state_params + k_x + k_tau + k_betas  #h_x, h_tau, beta

    inits = np.zeros(k)
    inits[:k_state_params] = params[:k_state_params] # state params
    inits[k_state_params:k_state_params+k_x] = 0. #x knots
    inits[k_state_params+k_x:k_state_params+k_x+k_tau] = params[k_state_params:k_state_params+1] #tau knots
    inits[k_state_params+k_x+k_tau:] = params[k_state_params+1:] #betas
    return inits

def profile_omega(params, profile_value):
    k = len(params)
    inits = np.zeros(k+1)
    inits[:4] = params[:4]
    inits[4:6] = params[4:6]
    inits[6] = profile_value
    inits[7:] = params[6:]
    return inits

def profile_rho(params, profile_value=float):
    k = len(params)
    inits = np.zeros(k+1)
    inits[:4] = params[:4]
    inits[4:7] = params[4:7]
    inits[7] = profile_value
    inits[8:] = params[7:]
    return inits


#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
picklefolder = os.path.join(basefolder, 'pickle_folder', )
HUNDRED = 100
days = 365
set_theme()
cm = diverge_map(high=colors[2], low=colors[0])

### CONFIG
ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
x_grid = np.array([-0.4, -0.3,  -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1,  0.2,  0.3, 0.4])
x_knots = np.array([-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
# tau_grid = np.linspace(0, 2, 49, endpoint=True).round(4)[1:]
tau_grid = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
tau_knots = np.array([6/12, 9/12, 12/12, 18/12])
cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])

vols_to_skip = dict({-0.4: np.array([0.0417, 0.0833, 0.125, 0.1667, 0.2083, 0.25]),
                     -0.3: np.array([0.0417, 0.0833, 0.125, 0.1667]),
                     -0.2: np.array([0.0417, 0.0833]),
                     # -0.1: np.array([0.0417]),
                     # 0.1: np.array([0.0417]),
                     0.2: np.array([0.0417, 0.0833]),
                     0.3: np.array([0.0417, 0.0833, 0.125, 0.1667]),
                     0.4: np.array([0.0417, 0.0833, 0.125, 0.1667, 0.2083, 0.25])})


# x_grid = np.array([-0.2, -0.15, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.15, 0.2])
# x_knots = np.array([-0.1, -0.05, 0.0, 0.05, 0.1,])
# # tau_grid = np.linspace(0, 2, 49, endpoint=True).round(4)[1:]
# tau_grid = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
# tau_knots = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12])
# cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])

df_vols_cube = import_ts_ivs(x_grid, tau_grid, ticker)

# for x in vols_to_skip.keys():
#     df_vols_cube.loc[df_vols_cube.CLOSEST_MONEYNESS == x, vols_to_skip[x]] = np.nan


df_vols_cube.iloc[:, 1:] **= 2
timeline = df_vols_cube.index.unique()

# df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS.isin([-0.4, 0.4])]

in_sample = (timeline <= '2023-07-31').sum()

tmp = df_vols_cube.set_index('CLOSEST_MONEYNESS')
tmp.isna().groupby(by=tmp.index).mean() * 100

df_vols_cube_avg = df_vols_cube[df_vols_cube.index.year.isin([2023])].groupby(by='CLOSEST_MONEYNESS').mean()
fig, ax= plt.subplots(figsize=(10,10))
m, n = df_vols_cube_avg.shape
for tau in df_vols_cube_avg.columns:
    ax.scatter(df_vols_cube_avg.index, df_vols_cube_avg[tau], color=cm(tau/2.5))

plt.tight_layout()
plt.show()

df_vols_cube_avg = df_vols_cube.loc[pd.to_datetime('2022-06-15')].groupby(by='CLOSEST_MONEYNESS').median()
# df_vols_cube_avg -= df_vols_cube_avg.mean(axis=1).values.reshape(13,1)
fig, ax= plt.subplots(figsize=(10,10))
m, n = df_vols_cube_avg.shape
for tau in df_vols_cube_avg.columns:
    ax.scatter(df_vols_cube_avg.index, df_vols_cube_avg[tau], color=cm(tau/2.5))

plt.tight_layout()
plt.show()

means = [
    # 'carr_wu_standard',
    # 'carr_wu_prop_fixed',
    # 'carr_wu_s1_fixed', 'carr_wu_s2_fixed',
    # 'carr_wu_prop_s1_dyn_fixed', 'carr_wu_prop_s2_dyn_fixed',
    # 'carr_wu_disprop_s1_dyn_fixed', 'carr_wu_disprop_s2_dyn_fixed',
    # 'carr_wu_prop_s1', 'carr_wu_prop_s2',
    # 'carr_wu_disprop_s1', 'carr_wu_disprop_s2', 'carr_wu_disprop_s3', 'carr_wu_disprop_s4',
    # 'carr_wu_disprop2_s4_time_space',
    # 'carr_wu_s4_decay_kappa_omega',
    # 'carr_wu_disprop2_sknots'
    'custom'
]

k_ar1_factors = [    
    # 0,
    # 0,
    # 0, 0,
    # 0, 0,
    # 0, 0,
    # 0, 0,
    # 0, 0, 0, 0,
    0, 0
    # 1,1,
    # 3, 3,
    # 4, 4,
]

variances = [
    'iid',
             'mvn-spline-restricted'
             ]

# df_vols_z_cube = df_vols_cube.copy()
# for tau in tau_grid:
#     # df_vols_z_cube[tau] = (df_vols_z_cube['CLOSEST_MONEYNESS'] + 0.5*(df_vols_z_cube[tau]) * tau) / (df_vols_z_cube[tau] * tau**.5)
#     df_vols_z_cube[tau] = (df_vols_z_cube['CLOSEST_MONEYNESS']) / (df_vols_z_cube[tau] * tau**.5)

tmp = df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS.isin([-0.2])] - df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS.isin([0.2])]
Iq = (tmp.abs().loc[:, 0.0417].ffill().bfill()**.5 - tmp.abs().loc[:, 0.25].ffill().bfill()**.5 ).abs()
fig, ax = plt.subplots(figsize=(20,10))
Iq.plot(ax=ax)
ax.axhline(y=0.1)
plt.tight_layout(), plt.show()
THRESHOLD = 0.05
Iq = (Iq <= THRESHOLD).values.astype(float)
ones_q = np.zeros(15)
ones_q[0] = 1
Sq = np.diag(ones_q)

# ### FIT MODELS
# for mean, k_ar1_factor in zip(means, k_ar1_factors):
#     print(mean)
#     for variance in variances:
#         print(variance)
#         k_h_sigmas = 1 if variance == 'iid' else len(x_knots) + len(tau_knots) - 1
#         h_sigmas_inits = np.ones(k_h_sigmas) * np.log(0.25**2)
#
#         inits = None
#         # if 'corr' in mean:
#         #     model_name = mean.replace('corr', f'{variance}.pkl')
#         #     dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, model_name))
#         #     transformed_params = dict_results['set_params'][-2].x
#         #     k, k_betas, _ = dict_results['param_dims']
#         #     inits = get_dyncorr_inits(transformed_params)
#         #
#         if variance == 'iid':
#             inits = None
#         else:
#             model_name = f'{mean}_iid.pkl'
#             dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, model_name))
#             transformed_params = dict_results['set_params'][-2].x
#             k_states, k_betas, k_params = dict_results['param_dims']
#             inits = get_mvn_from_iid_inits(transformed_params, k_states, k_ar1_factor, k_betas, x_knots, tau_knots)
#
#         model = DynamicSurfaceModels(df_surface=df_vols_cube)
#         model.specify_measurement_equation(mean=mean, variance=variance,
#                                            moneyness_pillars=x_knots, maturity_pillars=tau_knots,
#                                            cyclical_knots=cyclical_knots, k_ar1_factors=k_ar1_factor)
#
#         # if inits == None:
#         #     model.fit(N_fit=in_sample, cross_sectional=True)
#         #     f_hat, beta_hat, rmspe = model.f_cs, model.beta_cs, model._rmspe_cs
#         #     df_f_hat = pd.DataFrame(f_hat)
#         #     N, k = df_f_hat.shape
#         #     df_f_hat['CLOSEST_MONEYNESS'] = 0
#         #     t_end = str(pd.Timestamp.today().strftime(format='%Y/%m/%d'))
#         #     t_start = str((pd.Timestamp.today() - timedelta(days=N)).strftime(format='%Y/%m/%d'))
#         #     timeline = pd.date_range(start=t_start, end=t_end)
#         #     df_f_hat.index = timeline[:N]
#         #     Z = np.kron(np.ones((N, 1)), np.eye(k)).reshape(N, k, k)
#         #     lg_ssm = DynamicSurfaceModels(df_surface=df_f_hat)
#         #     lg_ssm.specify_measurement_equation(mean='linear', variance='diag', Z=Z, k_ar1_factors=k_ar1_factor)
#         #     lg_ssm.fit(find_filter_inits=False)
#         #     inits = np.concatenate((lg_ssm._mle._mle_optim_results.x[:-k], h_sigmas_inits))
#
#
#         model.fit(N_fit=in_sample, mle_inits=inits)
#         model.compute_stats()
#         pd.DataFrame(model.a[:, :, 0], index=model._time_line).plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
#         pd.DataFrame(model.rmspe, index=model._time_line).plot(figsize=(20, 10)), plt.tight_layout(), plt.show()
#
#         try:
#             plt.plot(model._builder.maturity_grid, model._builder._v_tau), plt.show()
#             plt.plot(model._builder.moneyness_grid, model._builder._u_x), plt.show()
#         except:
#             pass
#
#         model_name = f'{mean}_{variance}.pkl'
#         results_to_pickle = model._results_to_pickle()
#         dump(results_to_pickle, os.path.join(picklefolder, 'fitted_models', ticker, model_name))
#
#         # results_to_pickle = load(os.path.join(picklefolder, 'fitted_models', ticker, model_name))
#         # print(model_name, -np.round(207810*results_to_pickle['set_params'][-2].fun))
#         # 1+1




##### PROFILE LL eta3
model_name = 'carr_wu_disprop2_s4_mvn-spline.pkl'
mean = 'carr_wu_s4_decay_kappa_omega'
variance = 'mvn-spline'
dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
model = DynamicSurfaceModels(df_surface=df_vols_cube)
model.specify_measurement_equation(mean=mean, variance=variance,
                                   moneyness_pillars=x_knots, maturity_pillars=tau_knots,
                                   cyclical_knots=cyclical_knots
                                   )
optim_results = dict_results['set_params'][12]

M = 50
profile_grid_omega = np.linspace(-20, -5, M, endpoint=True)
ll_profile = np.zeros(M)
profile_a = np.zeros((in_sample, M))
for i in range(M):
    trans_params = profile_omega(optim_results.x, profile_value=profile_grid_omega[i])
    results = model.profile(N_fit=in_sample, transformed_params=trans_params)
    ll_profile[i] = results[0]
    profile_a[:, i] = results[1][:, 6, 0]
    print(i)
# transformed_params =
fig, ax = plt.subplots(figsize=(20*1.5,10*1.5), ncols=2)
ax[0].plot(profile_grid_omega, ll_profile, color='black')
ax[0].scatter(profile_grid_omega, ll_profile, c=cm(np.abs(profile_grid_omega/21)), s=200)
ax[0].axhline(y=579673, color='red', label='BASELINE LL')
for i in range(M):
    ax[1].plot(timeline[:in_sample], profile_a[:, i], color=cm(np.abs(profile_grid_omega[i]/21)))
ax[0].set_xlabel(r'$\log \sigma^2_{\eta_3}$', size=30)
ax[0].set_ylabel(r'Log-Lik', size=30)
ax[0].legend()
ax[0].set_title(r'Profile Log-Lik as a function of $\log \sigma^2_{\eta_3}$', size=35)
ax[1].set_title(r'Filtered paths for $\eta_{3,t}$ for various $\log \sigma^2_{\eta_3}$', size=35)
plt.tight_layout()
plt.show()
#
# for t in pd.DataFrame(profile_a, index=timeline[:in_sample]).diff().mean(axis=1).sort_values().iloc[:10].index.sort_values():
#     df_vols_cube_avg = df_vols_cube.loc[t].groupby(by='CLOSEST_MONEYNESS').median()
#     # df_vols_cube_avg -= df_vols_cube_avg.mean(axis=1).values.reshape(13,1)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     m, n = df_vols_cube_avg.shape
#     for tau in [0.0417, 0.0833, 0.125 , 0.1667]:
#         ax.scatter(df_vols_cube_avg.index, df_vols_cube_avg[tau], color=cm(tau/0.2), label=f'{tau}')
#     ax.legend()
#     ax.set_title(t)
#     plt.tight_layout()
#     plt.show()

#
#
# f_hat = pd.DataFrame(model.a_t[:, :, 0], index=model._time_line)
# for t in ['2020-07-29', '2021-03-11', '2021-06-11', '2021-10-11', '2021-12-13', '2022-06-13',  '2022-08-11']:
#     loc = np.argwhere(model._time_line == t).flatten()[0]
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
#     fit_ws = model._builder._Zf(f_ws, FIXED, model.betas, model._p, model._k).reshape(n, m).T **.5
#     fit_wos = model._builder._Zf(f_wos, FIXED, model.betas, model._p, model._k).reshape(n, m).T **.5
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


# # moneyness_grid = np.linspace(-.3, .3, 25, endpoint=True).round(4) # 0.025 step size
# # # maturity_grid = np.linspace(0, 25/24, 26, endpoint=True).round(4) #bi-weekly
# # maturity_grid = np.linspace(0, 49/24, 50, endpoint=True).round(4)
# # # maturity_grid = np.linspace(0, 2, 50, endpoint=True).round(4)
# # moneyness = np.array([-0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2])
# # # maturities = np.linspace(0, 25/24, 26, endpoint=True).round(4)[2:]
# # maturities = np.linspace(0, 49/24, 50, endpoint=True).round(4)[2:]
# # cols = ['CLOSEST_MONEYNESS'] + maturities.tolist()
# # filtered_dir = 'C:/Users/PI26UT/Documents/Data/ComdtyVols/Filtered Data'
# # tickers = [
# #     # 'Crude Futures_Brent 1st Line_BRENT',
# #            'NG Pen Futures 25k ICE Lots_Henry_HH',
# #           # 'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH'
# # ]
# #
# # maturity_pillars = np.array([2/24, 3/24, 4/24, 6/24, 8/24, 12/24, 18/24, 1])
# # moneyness_pillars = np.array([#-0.2, -0.1,
# #                             # -0.15,  -0.035, -0.015,
# #     -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15
# #     # 0.015, 0.035, 0.15
# #                               # 0.1, 0.2
# #                               ])
# # moneyness_pillars = moneyness #/1.1
# # for ticker in tickers:
# #     try:
# #         df_vols_cube = load(os.path.join(picklefolder, f'iv_ts_cube_{ticker}.pkl'))
# #     except:
# #         df_vols = pd.read_csv(os.path.join(filtered_dir, f'{ticker}.csv'), sep=';')
# #         df_vols_cube = build_ts_cube(df_vols, maturity_grid, moneyness_grid)
# #         dump(df_vols_cube, os.path.join(picklefolder, f'iv_ts_cube_{ticker}.pkl'))
# #
# #     df_vols_cube = df_vols_cube[cols]
# #     df_vols_cube = df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS.isin(moneyness)]
# #
# #     df_vols_cube.iloc[:, 1:] **= 2
# #     model = DynamicSurfaceModels(df_surface=df_vols_cube)
# #     model.specify_measurement_equation(mean='carr_wu_seasonal_term_structure_dynamic', variance='iid',
# #                                        moneyness_pillars=moneyness_pillars, maturity_pillars=maturity_pillars)
#     # model.fit(collapse=True)
#     # # inits0 = np.random.normal(model._mle._mle_optim_results.x*0.99, scale=0.01)
#     # # model.fit(mle_inits=inits0)
#     #
#     # model.compute_stats()
#     # labels = [r'$\mu_t$', r'$\log(\omega_t)$', r'$\log(v_t)$',  r'$\log\left(\frac{1+\rho^{min}_t}{1-\rho^{min}_t}\right)$',
#     #           r'$\log\left(\frac{1+\rho^{max}_t}{1-\rho^{max}_t}\right)$',
#     #           r'$\delta_{1,t}$', r'$\delta^*_{1,t}$', r'$\delta_{2,t}$', r'$\delta^*_{2,t}$',
#     #           r'$\delta_{3,t}$', r'$\delta^*_{3,t}$',]
#     # fig, ax = plt.subplots(figsize=(20,10), ncols=3, nrows=3, sharex=True)
#     # axs = ax.flatten()
#     # for i in range(4):
#     #     lb = (model.a_t[:, i, 0] - 2*model.P_t[:, i, i]**.5).flatten()[30:]
#     #     ub = (model.a_t[:, i, 0] + 2*model.P_t[:, i, i]**.5).flatten()[30:]
#     #     axs[i].fill_between(model._time_line[30:], lb, ub, color=colors[i], label='_nolegend_', alpha=0.5)
#     #     pd.DataFrame(model.a_t[:, i], index=model._time_line, columns=[labels[i]]).iloc[30:].plot(ax=axs[i], legend=False, color=colors[i])
#     #
#     # pd.DataFrame(model.rmspe, index=model._time_line, columns=['RMSPE']).iloc[30:].plot(ax=axs[-1], legend=False, color=colors[5])
#     # # pd.DataFrame(model.mae, index=model._time_line, columns=['MAE']).iloc[30:].rolling(5).mean().plot(ax=axs[-1], legend=False)
#     #
#     # for i in range(9):
#     #     axs[i].axhline(y=0, color=colors[5], label='_nolegend_')
#     #     axs[i].set_xlabel('')
#     #     axs[i].legend()
#     #     axs[i].set_xlim(model._time_line[30:].min(), model._time_line.max())
#     #
#     # plt.tight_layout()
#     # plt.show()
#
#
#     model.fit(cross_sectional=True)
#
#     for t in ['2020-07-29', '2021-03-11', '2021-06-11', '2021-10-11', '2021-12-13', '2022-06-13',  '2022-08-11']:
#         loc = np.argwhere(model._time_line == t).flatten()[0]
#         f_ws = pd.DataFrame(model.a_t[:, :, 0], index=model._time_line).iloc[loc].values.reshape(model._k,1)
#         f_wos = f_ws.copy()
#         f_wos[4:] = 0
#         t0 = model.day_of_year(pd.to_datetime(t))
#         tmp = model.df_surface.loc[t].sort_values(by='CLOSEST_MONEYNESS')
#         Y = tmp.drop(columns='CLOSEST_MONEYNESS').copy().values **.5
#         m, n = Y.shape
#         x = tmp.CLOSEST_MONEYNESS.values.reshape(m, 1) * np.ones((m, n))
#         tau = tmp.drop(columns='CLOSEST_MONEYNESS').columns.values.reshape(n, 1).T * np.ones((m, n))
#         T = t0 + tau
#         FIXED = List()
#         FIXED.append(model.vec(x))
#         FIXED.append(model.vec(tau))
#         FIXED.append(model.vec(T))
#         FIXED.append(np.eye(5))
#
#         fit_ws = model._builder._Zf(f_ws, FIXED, model.betas, model._N, model._m).reshape(n, m).T **.5
#         fit_wos = model._builder._Zf(f_wos, FIXED, model.betas, model._N, model._m).reshape(n, m).T **.5
#
#         fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
#         i = 0
#         for i in range(m):
#             ax[0].scatter(tau[i], Y[i], color=cm(np.linspace(0, 1, m)[i]))  # , label=list_T[i])
#             ax[0].plot(tau[i], fit_ws[i], linestyle='dashed', color=cm(np.linspace(0, 1, m)[i]), label=f'x={moneyness[i]}')
#             i += 1
#
#         i = 0
#         for tau in maturities:
#             ax[1].plot(x[:, i], fit_wos[:, i], linestyle='dashed', color=cm(tau), label=f'tau={tau}')
#             i += 1
#
#         ax[0].set_xlabel('Maturity (Years)')
#         ax[1].set_xlabel('Log-Moneyness')
#         ax[0].legend(ncol=3, prop={'size': 15})
#         ax[1].legend(ncol=3, prop={'size': 15})
#         plt.tight_layout()
#         plt.show()
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