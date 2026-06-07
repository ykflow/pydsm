import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotting_tools.set_plotting_theme import set_theme, colors, diverge_map
import matplotlib.pyplot as plt
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from config_utils.config_utils_main import import_ts_ivs
from calculator.forward_curve_calculator import ForwardCurveCalculator
from calculator.yield_curve_calculator import ZeroRatesCalculator
from models.static.surface_models.seasonal_ssvi_model import SeasonalSurfaceSVIModel
from numba.typed import List
from pricers.variance_swap_rate import VarianceSwapPricer
from joblib import dump, load


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

fcc = ForwardCurveCalculator(os.path.join(datafolder, 'comdty_forwards'))
zrc = ZeroRatesCalculator(os.path.join(datafolder, 'rates'))

df_forwards = fcc.interpolate(tau_grid, timeline)
df_rates = zrc.interpolate(tau_grid, timeline)

vsp = VarianceSwapPricer(df_vols_cube, df_forwards, df_rates)
vsp.surface_interpolator()
vsp.compute_variance_swap_rates()


## LOAD MODEL
mean = 'carr_wu_disprop2_s4'
variance = 'mvn-spline'
model_name = f'{mean}_{variance}.pkl'
dict_results = load(os.path.join(picklefolder, 'fitted_models', ticker, 'mle_objects', model_name))
model = DynamicSurfaceModels(df_surface=df_vols_cube[df_vols_cube.index <=train_end])
model.specify_measurement_equation(mean=mean, variance=variance,
                                   moneyness_pillars=x_knots, maturity_pillars=tau_knots,
                                   cyclical_knots=cyclical_knots)
model._set_model_params(*dict_results['set_params'])
try:
    model.run_filter()
except:
    model.run_filter()
model.compute_stats()


vecX = np.array([[0.], [0.]], dtype=float)
vecTAU = np.array([[0.0833], [0.25]], dtype=float)
model.run_simulation_smoother(MC=5000, vecX=vecX, vecTAU=vecTAU)

labels = [r"$I_t(0,$1M)", r"$I_t(0,$3M)"]
y_sim = model.y_sim.copy()
t_max = df_vols_cube.index.max()
list_y_sim = []
list_cols = []
for i in range(2):
    df_atm = vsp.variance_surfaces[vsp.variance_surfaces.CLOSEST_MONEYNESS == 0][vecTAU.flatten()[i]]
    y_sim_i = pd.DataFrame(y_sim[:, i, :], index=model._time_line_sim).iloc[2:]
    lb, ub = df_atm.min(), df_atm.max()

    y_sim_i[(y_sim_i.loc[y_sim_i.index > train_end] > ub * 8)] = np.nan
    y_sim_i[(y_sim_i.loc[y_sim_i.index > train_end] < lb / 8)] = np.nan
    y_sim_i = y_sim_i.dropna(axis=1)
    print(y_sim_i.shape)
    list_y_sim.append(y_sim_i **.5 * HUNDRED)
    list_cols.append(y_sim_i.columns)

intersect = np.intersect1d(*list_cols)[:1000]
print(len(intersect))
crls = [colors[0], colors[2]]
scenario = 74
for idx in intersect[25:50]:
    fig, ax = plt.subplots(figsize=(20, 10))
    for i in range(2):
        df_atm = vsp.variance_surfaces[vsp.variance_surfaces.CLOSEST_MONEYNESS == 0][vecTAU.flatten()[i]] **.5 * HUNDRED
        y_sim_i = list_y_sim[i][intersect]
        print(y_sim_i.shape)

        lb_i = y_sim_i.quantile(1-0.90, axis=1)
        ub_i = y_sim_i.quantile(0.90, axis=1)
        ax.plot(lb_i[lb_i.index >= train_end], color=crls[i], linestyle='dotted', label="_nolegend_",alpha=0.5)
        ax.plot(ub_i[ub_i.index >= train_end], color=crls[i], linestyle='dotted', label="_nolegend_",alpha=0.5)

        lb_i[(lb_i.index > train_end) & (lb_i.index <= t_max)] = np.nan
        ub_i[(ub_i.index > train_end) & (ub_i.index <= t_max)] = np.nan

        y_sim_i_med = y_sim_i.mean(axis=1)
        y_sim_i_med.loc[train_end:t_max] = np.nan
        # y_sim_i_med.loc[:t_max] = np.nan

        ax.scatter(df_atm.index, df_atm.values, color=crls[i], label=f"{labels[i]}", s=4)
        y_sim_i_med.plot(ax=ax, legend=False, color=crls[i], label=f"MC Avg: {labels[i]}", linewidth=1)
        ax.fill_between(lb_i.index, lb_i, ub_i, color=crls[i], alpha=0.5, label=f"MC CI(90%): {labels[i]}")
        (y_sim_i.loc[train_end:t_max, scenario]).plot(ax=ax, legend=False, color=["blue", "red"][i],  label=f"Scenario 1: {labels[i]}", linewidth=1)
        # (y_sim_i.loc[ :t_max, scenario]).plot(ax=ax, legend=False, color=["blue", "red"][i],
        #                                         label=f"Scenario 1: {labels[i]}", linewidth=1, alpha = 0.5)

    ax.text(x=pd.to_datetime('20220920'), y=162, s=r'In-sample $(N=759)$', size=20)
    ax.text(x=pd.to_datetime('20230901'), y=162, s=r'Out-of-sample $(h=276)$', size=20)
    ax.text(x=pd.to_datetime('20241101'), y=162, s=r'Future $(h=504)$', size=20)
    ax.axvline(x=pd.to_datetime(train_end), color=colors[5], label="_nolegend_")
    ax.axvline(x=pd.to_datetime(t_max), color=colors[5], label="_nolegend_")
    ax.set_ylim(10, 180)
    ax.set_xlim(min(model._time_line_sim), max(model._time_line_sim))
    ax.legend(ncols=2, scatterpoints=5,  prop={'size': 18})
    ax.set_ylabel(r'%')
    # ax.set_title(idx, size=25)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(plotfolder, 'pfe.pdf'))



# t = pd.to_datetime('20210726')
# loc = np.argwhere(timeline == t).item(0)
# fig, ax = plt.subplots(figsize=(21,14), ncols=3, nrows=2, subplot_kw=dict(projection="3d"))
# axs = ax.flatten()
# for ax_ in axs:
#     ax_.view_init(azim=45, elev=25)
#
# fit_ws = model.df_fitted_vols.loc[t].values[:, 1:] *HUNDRED
# Y = model.df_vols.loc[t].values[:, 1:] * HUNDRED
# axs[0].scatter(model.vec(model.X), model.vec(model.TAU), model.vec(Y), color=colors[2], label=r'$I_t$')
# axs[0].plot_surface(model.X, model.TAU, fit_ws, color=colors[0], alpha=0.5, label=r'$\widehat{I}_{t|t}$', linewidth=1/2)
# axs[0].set_xlim(x_grid.max(), x_grid.min())
#
# f_sim_smoothed = model.a_sim[loc]
# vecT = model.day_of_year(model._time_line[loc]) + model.vecTAU
# FIXED = List()
# FIXED.append(model.vecX)
# FIXED.append(model.vecTAU)
# FIXED.append(vecT)
# FIXED.append(model.vecB)
# y_sim = model._builder.Zf(f_sim_smoothed.mean(axis=1), FIXED, model.betas, model._p, model._k) ** .5 *HUNDRED
# Y_sim = model.unvec(y_sim, model._m, model._n)
# axs[1].scatter(model.vec(model.X), model.vec(model.TAU), model.vec(Y), color=colors[2], label=r'$I_t$')
# axs[1].plot_surface(model.X, model.TAU, Y_sim, color=colors[0], alpha=0.5,  label=r'$\widehat{I}_{t|t}$', linewidth=1/2)
# axs[1].set_xlim(x_grid.max(), x_grid.min())
#
# perc = [80, 40, 60, 10]
# j = 0
# for i in range(2, 6):
#     y_sim = model._builder.Zf(np.percentile(f_sim_smoothed, q=perc[j], axis=1), FIXED, model.betas, model._p, model._k) ** .5 * HUNDRED
#     Y_sim = model.unvec(y_sim, model._m, model._n)
#     axs[i].scatter(model.vec(model.X), model.vec(model.TAU), model.vec(Y), color=colors[2], label=r'$I_t$',linewidth=1/2)
#     axs[i].plot_surface(model.X, model.TAU, Y_sim, color=colors[0], alpha=0.5,
#                           label=r'$\widehat{I}_{t|t}$')
#     axs[i].set_xlim(x_grid.max(), x_grid.min())
#     j +=1
#
# titles = ['Filtered', 'Smoothed', 'Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4']
# for i in range(6):
#     axs[i].set_title(titles[i], size=35)
#     axs[i].set_zlim(20, 90)
#     axs[i].set_xlabel(r'$x$', labelpad=10)
#     axs[i].set_ylabel(r'$\tau$', labelpad=10)
#     axs[i].zaxis.set_rotate_label(False)
#     axs[i].set_zlabel(r'$I_t(x, \tau)$ (%)', rotation=90)
# plt.tight_layout()
# # plt.show()
# plt.savefig(os.path.join(plotfolder, 'smoothed_scenarios.pdf'))