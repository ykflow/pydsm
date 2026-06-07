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
from numba.typed import List
from joblib import dump, load
from statistical_tests.model_confidence_set_test import ModelConfidenceSet as MCS


def truncate(x, decimals=3):
    return np.floor(x * 10 ** decimals) / 10 ** decimals

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

means = [
    # 'carr_wu_standard',
    # 'carr_wu_prop_fixed',
    # 'carr_wu_s1_fixed', 'carr_wu_s2_fixed',
    'carr_wu_prop_s1_dyn_fixed', 'carr_wu_prop_s2_dyn_fixed',
    'carr_wu_disprop_s1_dyn_fixed', 'carr_wu_disprop_s2_dyn_fixed',
    'carr_wu_prop_s1', 'carr_wu_prop_s2',
    'carr_wu_disprop_s1',
    'carr_wu_disprop_s2',
    'carr_wu_disprop_s3',
    'carr_wu_disprop_s4',
    'carr_wu_disprop2_s3',
    'carr_wu_disprop2_s4',
]

models_iid = [f'{x}_iid' for x in means]
models_mxn = [f'{x}_mvn-spline' for x in means]
model_names = models_iid + models_mxn + ['ssvi']

### CONFIG
ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
x_grid = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
x_knots = np.array([-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
# tau_grid = np.linspace(0, 2, 49, endpoint=True).round(4)[1:]
tau_grid = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
tau_knots = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12])
cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])
train_end = '20230731'

## DATA IMPORTS
df_vols_cube = import_ts_ivs(x_grid, tau_grid, ticker)
df_vols_cube = df_vols_cube[(df_vols_cube.index >= start) & (df_vols_cube.index <= end)]
df_vols_cube.iloc[:, 1:] **= 2
timeline = df_vols_cube.index.unique()
in_sample = ((timeline <= train_end).sum()).astype(int)

x_bins = dict({-0.4: 'DITM', -0.3: 'DITM',
               -0.2: 'ITM', -0.1: 'ITM',
                -0.05: 'ATM', -0.025: 'ATM',  0.0:'ATM',  0.025:'ATM', 0.05:'ATM',
                0.1:'OTM',  0.2: 'OTM',
               0.3:'DOTM', 0.4:'DOTM'})
tau_bins = pd.cut(tau_grid,  bins=[0, 0.0833, 0.25, 0.5, 1, 2.5]).unique().tolist()
x_bins_labels = ['DITM', 'ITM', 'ATM', 'OTM', 'DOTM']
horizons = [1, 5, 21, 63, 126, 252]

dict_forecast_cs_rmspe = dict()
dict_forecast_errors = dict()
### FIT MODELS
for h in horizons:
    tmp1 = dict()
    tmp2 = dict()
    for model_name in model_names:
        print(model_name)
        forecast_objects = load(os.path.join(picklefolder, 'fitted_models', ticker, 'forecast_errors', f'{model_name}.pkl'))
        rmspe, df_errors = forecast_objects[0][h], forecast_objects[1][h]

        df_errors = df_errors.reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS'])
        df_errors.columns = pd.cut(df_errors.columns,  bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
        df_errors = df_errors.reset_index().set_index(['DATE'])
        df_errors[['CLOSEST_MONEYNESS']] = df_errors[['CLOSEST_MONEYNESS']].replace(x_bins).astype(str)
        tmp1[model_name] = pd.DataFrame(rmspe, index=timeline, columns=[model_name])
        tmp2[model_name] = df_errors
    dict_forecast_cs_rmspe[h] = tmp1
    dict_forecast_errors[h] = tmp2


dict_avg_forecast_errors_h_x_tau = dict()
for h in horizons:
    dict_avg_errors_x_tau = dict()
    dict_errors = dict_forecast_errors[h]
    for tau in tau_bins:
        for x in x_bins_labels:
            df = pd.DataFrame()
            for key in dict_errors.keys():
                df_errors = dict_errors[key]
                df_errors_bucket = df_errors[df_errors.CLOSEST_MONEYNESS == x][tau]
                cols = np.arange(df_errors_bucket.shape[1]).tolist()
                df_errors_bucket.columns = cols
                df_errors_bucket = pd.melt(df_errors_bucket.reset_index(), id_vars='DATE', value_vars=cols, value_name=key).drop(columns=['variable'])
                df_avg_errors = df_errors_bucket.groupby(by='DATE').mean()
                df = pd.concat([df, df_avg_errors], axis=1)
            dict_avg_errors_x_tau[f'{x}_{tau}'] = df
    dict_avg_forecast_errors_h_x_tau[h] = dict_avg_errors_x_tau


for h in horizons:
    df_mspe = pd.DataFrame()
    dict_cs_rmpse = dict_forecast_cs_rmspe[h]
    for key in dict_cs_rmpse.keys():
        df_mspe = pd.concat([df_mspe, (dict_cs_rmpse[key]/HUNDRED)**2], axis=1)
    dict_avg_forecast_errors_h_x_tau[h]['Avg.'] = df_mspe


df_counts_train = pd.DataFrame()
df_counts_test = pd.DataFrame()
dict_acc_train = dict()
dict_acc_test = dict()
for h in horizons:
    alpha, w, B = 0.05, 252, 1000
    dict_avg_errors_x_tau = dict_avg_forecast_errors_h_x_tau[h]
    df_mcs_pvalue_train = pd.DataFrame(columns=dict_avg_errors_x_tau.keys(), index=model_names)
    df_mcs_included_train = pd.DataFrame(columns=dict_avg_errors_x_tau.keys(), index=model_names)
    df_mcs_pvalue_test = pd.DataFrame(columns=dict_avg_errors_x_tau.keys(), index=model_names)
    df_mcs_included_test = pd.DataFrame(columns=dict_avg_errors_x_tau.keys(), index=model_names)

    df_accuracy_train = pd.DataFrame(index=model_names)
    df_accuracy_test = pd.DataFrame(index=model_names)
    dict_avg_errors_x_tau = dict_avg_forecast_errors_h_x_tau[h]
    for key in dict_avg_errors_x_tau.keys():
        print(key)
        df = dict_avg_errors_x_tau[key]
        df_train = df.iloc[10:in_sample].dropna()
        df_test = df.iloc[in_sample:].dropna()
        df_accuracy_train = pd.concat([df_accuracy_train, pd.DataFrame(df_train.mean()).rename(columns={0: key}) **.5 *HUNDRED], axis=1)
        df_accuracy_test = pd.concat([df_accuracy_test, pd.DataFrame(df_test.mean()).rename(columns={0: key}) **.5 *HUNDRED], axis=1)

        mcs = MCS(df_train, alpha, w, B)
        mcs.run()

        for included in mcs.included:
            df_mcs_included_train.loc[included, key] = '$^{\star}$'
        df_mcs_pvalue_train.loc[:, key] = r"\tiny{[" + mcs.pvalues.loc[model_names].apply(lambda x: truncate(x)).astype(str) + ']}'

        mcs = MCS(df_test, alpha, w, B)
        mcs.run()
        for included in mcs.included:
            df_mcs_included_test.loc[included, key] = '$^{\star}$'
        df_mcs_pvalue_test.loc[:, key] = r"\tiny{[" + mcs.pvalues.loc[model_names].apply(lambda x: truncate(x)).astype(str) + ']}'

    df_mcs_included_train.fillna('$^{}$', inplace=True)
    df_mcs_included_test.fillna('$^{}$', inplace=True)
    df_accuracy_train = df_accuracy_train.round(3).astype(str) + df_mcs_included_train
    df_accuracy_test = df_accuracy_test.round(3).astype(str) + df_mcs_included_test

    df_mcs_included_train

    count_train = pd.DataFrame(df_accuracy_test.agg(lambda x: x.str.contains('star')).sum(1), columns=['Count']).astype(int)
    count_test = pd.DataFrame(df_accuracy_test.agg(lambda x: x.str.contains('star')).sum(1), columns=['Count']).astype(int)
    df_counts_train = pd.concat([df_counts_train, count_train], axis=1)
    df_counts_test = pd.concat([df_counts_test, count_test], axis=1)

    df_accuracy_train = pd.concat([df_accuracy_train, count_train], axis=1)
    df_accuracy_test = pd.concat([df_accuracy_test, count_train], axis=1)

    df_accuracy_train = pd.concat([df_accuracy_train, df_mcs_pvalue_train], axis=0).loc[model_names]
    df_accuracy_test = pd.concat([df_accuracy_test, df_mcs_pvalue_test], axis=0).loc[model_names]
    df_accuracy_train.to_latex(os.path.join(tablefolder, f'mcs_in_sample_h={h}.tex'))
    df_accuracy_test.to_latex(os.path.join(tablefolder, f'mcs_out_of_sample_h={h}.tex'))

    dict_acc_train[h] = df_accuracy_train
    dict_acc_test[h] = df_accuracy_test



def gray(x):
    if '$^{\star}$'in str(x):
        tmp = x.replace('$^{\star}$', '')
        tmp = "\cellcolor[HTML]{9B9B9B}" + tmp
        return tmp
    else:
        return x


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




list_results = []
for h in horizons:
    df = dict_acc_test[h]
    df_results_tmp = pd.DataFrame()
    for model_name in [
                       'carr_wu_disprop_s2_dyn_fixed_mvn-spline',
        'carr_wu_disprop2_s4_iid',
                       'carr_wu_disprop2_s4_mvn-spline',
                       'ssvi']:
        print(model_name)
        rmspe = df.loc[model_name].iloc[0]
        pvals = df.loc[model_name].iloc[1]

        rmspe = pd.DataFrame(rmspe[:-2].values.reshape(5, 5).T, index=x_bins_labels)
        pvals = pd.DataFrame(pvals[:-2].values.reshape(5, 5).T, index=x_bins_labels)
        rmspe_float = rmspe.copy()
        pvals_float = pvals.copy()
        for i in range(5):
            rmspe_float[i] = rmspe_float[i].str.replace('$^{}$', '')
            rmspe_float[i] = rmspe_float[i].str.replace('$^{\star}$', '')
            pvals_float[i] = pvals_float[i].str.replace(r'\tiny{[', '')
            pvals_float[i] = pvals_float[i].str.replace(r']}', '')
        rmspe_float = rmspe_float.astype(float)
        pvals_float = pvals_float.astype(float)



        tmp = pvals_float.map(cell_color) + rmspe_float.astype(str)
        tmp = tmp.astype(str)
        counts = tmp.agg(lambda x: x.str.contains(r'cellcolor')).sum(1)
        tot = [r"\tiny{" + f"({x})" + r"}" for x in counts]
        pvals['Avg.'] = tot
        tmp['Avg.'] = rmspe_float.mean(axis=1).round(3)

        summary = pd.concat([tmp, pvals], axis=0, ignore_index=False).loc[x_bins_labels]
        summary.loc['Avg.'] = pd.concat([rmspe_float.mean(axis=0), pd.Series(rmspe_float.mean(axis=0).mean())]).values.round(3)

        df_results_tmp = pd.concat([df_results_tmp, summary], axis=1)

    list_results.append(df_results_tmp)

df_results = pd.concat(list_results, axis=0)
df_results.to_latex(os.path.join(tablefolder, f'mcs_summary_out_of_sample.tex'))




model_names = ['carr_wu_disprop_s2_dyn_fixed_iid',  'carr_wu_disprop_s2_dyn_fixed_mvn-spline', 'carr_wu_disprop_s2_iid',
               'carr_wu_disprop_s2_mvn-spline', 'carr_wu_disprop2_s4_iid', 'carr_wu_disprop2_s4_mvn-spline']
fig, ax = plt.subplots(figsize=(20,10))
ax.axvline(x=pd.to_datetime(train_end), linestyle='dotted', color=colors[5], label='_nolegend_')
ax.plot(dict_avg_forecast_errors_h_x_tau[1]['Avg.'][model_names]**.5 * HUNDRED)
ax.set_xlim(timeline.min(), timeline.max())
ax.set_ylabel('RMSPE (%)')
ax.legend([r'CW-disprop$_1$-S(2)-iid', r'CW-disprop$_1$-S(2)-MXVN', r'CW-disprop$_1$-SS(2)-iid', 'CW-disprop$_1$-SS(2)-MXVN', 'CW-disprop$_2$-SS(4)-iid', 'CW-disprop$_2$-SS(4)-MXVN'])
ax.text(x=pd.to_datetime('20230203'), y=16, s=r'In-sample', size=25)
ax.text(x=pd.to_datetime('20230901'), y=16, s=r'Out-of-sample', size=25)
ax.set_ylim(0, 40)
plt.tight_layout()
plt.savefig(os.path.join(plotfolder, 'daily_rmspe.pdf'))
# plt.show()


df_stats_train = pd.DataFrame()
df_stats_test = pd.DataFrame()
for model_name in model_names:
    df_stats_tmp = load(os.path.join(picklefolder, 'fitted_models', ticker, 'statistics', f'{model_name}.pkl')).reset_index()
    df_stats_tmp.index = [model_name]* len(df_stats_tmp.index)
    df_stats_tmp_train = df_stats_tmp.iloc[:5]
    df_stats_tmp_test = df_stats_tmp.iloc[5:]
    df_stats_train = pd.concat([df_stats_train, df_stats_tmp_train], axis=0)
    df_stats_test = pd.concat([df_stats_test, df_stats_tmp_test], axis=0)

list_df = []
for df in [df_stats_train,
           # df_stats_test
           ]:
    df = df.reset_index()
    p_cols = df.columns[2:]
    for col in p_cols:
        df[col] = truncate(df[col])
    for metric in df['index'].unique():
        for col in p_cols:
            tmp = df.loc[df['index'] == metric,col]
            if metric != 'R2':
                optimum = tmp == tmp.min()
            else:
                optimum = tmp == tmp.max()
            idx = optimum[optimum == True].index.item()
            value = str(df.loc[idx, col])
            df.loc[idx, col] = r'\textbf{' + value + r'}'

    list_df.append(df)

df_stats = pd.concat(list_df, axis=0)
df_stats.set_index('level_0').to_latex(os.path.join(tablefolder, 'summary_daily_metrics.tex'))

df_list = []
for model_name in model_names:
    df_data = load(os.path.join(picklefolder, 'fitted_models', ticker, 'squared_relative_errors', f'{model_name}.pkl'))
    # df_data = df_data[df_data.index > train_end]
    df_data = df_data.reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS'])
    tau_bins = pd.cut(df_data.columns, bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
    df_data.columns = tau_bins
    df_data = df_data.reset_index().set_index(['DATE'])
    df_data[['CLOSEST_MONEYNESS']] = df_data[['CLOSEST_MONEYNESS']].replace(x_bins).astype(str)

    df_avg = pd.DataFrame()

    for tau in tau_bins:
        for x in x_bins_labels:
            df_bucket = df_data[df_data.CLOSEST_MONEYNESS == x][tau].values.flatten()
            df_avg.loc[x, tau] = np.sqrt(np.nanmean(df_bucket)) * HUNDRED

    df_list.append(df_avg)

df_bucketted_errors = pd.concat([pd.concat(df_list[:3], axis=1), pd.concat(df_list[3:], axis=1)], axis=0)
df_bucketted_errors.style.format(decimal='.', thousands=',', precision=3).to_latex(os.path.join(tablefolder, 'bucketted_rmspe.tex'))

################################################
model_names = ['carr_wu_disprop_s2_dyn_fixed_iid',  'carr_wu_disprop_s2_dyn_fixed_mvn-spline', 'carr_wu_disprop_s2_iid',
               'carr_wu_disprop_s2_mvn-spline', 'carr_wu_disprop2_s4_iid', 'carr_wu_disprop2_s4_mvn-spline']
x_grid = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
x_knots = np.array([-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
# tau_grid = np.linspace(0, 2, 49, endpoint=True).round(4)[1:]
tau_grid = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
tau_knots = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12])
cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])

## DATA IMPORTS
df_vols_cube = import_ts_ivs(x_grid, tau_grid, ticker)
df_vols_cube = df_vols_cube[(df_vols_cube.index >= start) & (df_vols_cube.index <= end)]
timeline = df_vols_cube.index.unique()
is_na = df_vols_cube.isna()
df_list = []
fig, ax = plt.subplots(figsize=(20,10))
ax.axvline(x=pd.to_datetime(train_end), linestyle='dotted', color=colors[5], label='_nolegend_')
for model_name in [ 'carr_wu_disprop2_s4_iid', 'carr_wu_disprop2_s4_mvn-spline']:
    df_vols_fitted = load(os.path.join(picklefolder, 'fitted_models', ticker, 'fitted_vols', f'{model_name}.pkl'))
    df_vols_fitted[is_na] = np.nan
    df_true = df_vols_cube.copy()

    df_true = df_true.reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS'])
    tau_bins = pd.cut(df_true.columns, bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
    df_true.columns = tau_bins
    df_true = df_true.reset_index().set_index(['DATE'])
    df_true[['CLOSEST_MONEYNESS']] = df_true[['CLOSEST_MONEYNESS']].replace(x_bins).astype(str)

    df_vols_fitted = df_vols_fitted.reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS'])
    tau_bins = pd.cut(df_vols_fitted.columns, bins=[0, 0.0833, 0.25, 0.5, 1, 2.5])
    df_vols_fitted.columns = tau_bins
    df_vols_fitted = df_vols_fitted.reset_index().set_index(['DATE'])
    df_vols_fitted[['CLOSEST_MONEYNESS']] = df_vols_fitted[['CLOSEST_MONEYNESS']].replace(x_bins).astype(str)

    df_avg = pd.DataFrame()


    df_skew_true_tau = df_true[df_true.CLOSEST_MONEYNESS == 'DOTM'].iloc[:, 1:] - df_true[df_true.CLOSEST_MONEYNESS == 'DITM'].iloc[:, 1:]
    df_skew_fitted_tau = df_vols_fitted[df_vols_fitted.CLOSEST_MONEYNESS == 'DOTM'].iloc[:, 1:] - df_vols_fitted[df_vols_fitted.CLOSEST_MONEYNESS == 'DITM'].iloc[:, 1:]

    df_skew_true_tau = df_skew_true_tau.groupby(by=df_skew_true_tau.index).mean().mean(axis=1)
    df_skew_fitted_tau = df_skew_fitted_tau.groupby(by=df_skew_fitted_tau.index).mean().mean(axis=1)

    df_list.append(df_skew_fitted_tau)

    ax.plot(timeline, df_skew_fitted_tau.values *HUNDRED)

ax.plot(timeline, df_skew_true_tau.values*HUNDRED, linestyle='dotted', color='black')
ax.set_xlim(timeline.min(), timeline.max())
ax.set_ylabel('Skew (%)')
ax.legend([ 'CW-disprop$_2$-SS(4)-iid', 'CW-disprop$_2$-SS(4)-MXVN', 'True'], loc='lower left', ncols=2)
ax.text(x=pd.to_datetime('20230203'), y=15, s=r'In-sample', size=25)
ax.text(x=pd.to_datetime('20230901'), y=15, s=r'Out-of-sample', size=25)
# ax.set_ylim(0, 40)
plt.tight_layout()
plt.savefig(os.path.join(plotfolder, 'daily_skew.pdf'))
# plt.show()
