import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from plotting_tools.set_plotting_theme import set_theme, colors
from monte_carlo.monte_carlo_expirements import MonteCarloExperiments
from models.dynamic.dynamic_surface_models import DynamicSurfaceModels
from data_utils.build_ts_cubes import build_ts_cube
from joblib import dump, load

#### CONFIG
codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
basefolder = codefolder.parent
datafolder = os.path.join(basefolder, 'data_storage', )
plotfolder = os.path.join(basefolder, 'plot_folder', )
picklefolder = os.path.join(basefolder, 'pickle_folder', )
set_theme()

moneyness_grid = np.linspace(-.5, .5, 41, endpoint=True).round(4) # 0.025 step size
maturity_grid = np.linspace(0, 49/24, 50, endpoint=True).round(4)
moneyness = np.array([ -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2])
maturities = np.linspace(0, 49/24, 50, endpoint=True).round(4)[2:]
cols = ['CLOSEST_MONEYNESS'] + maturities.tolist()
filtered_dir = 'C:/Users/PI26UT/Documents/Data/ComdtyVols/Filtered Data'
tickers = [
    # 'Crude Futures_Brent 1st Line_BRENT',
           'NG Pen Futures 25k ICE Lots_Henry_HH',
          # 'Dutch TTF Gas Base Load Futures_TTF_TTF_MAH'
]

maturity_pillars = np.array([2/24, 3/24, 4/24, 6/24, 8/24, 12/24, 18/24, 1])
moneyness_pillars = np.array([#-0.2, -0.1,
                            # -0.15,  -0.035, -0.015,
    -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15
    # 0.015, 0.035, 0.15
                              # 0.1, 0.2
                              ])
moneyness_pillars = moneyness #/1.1
cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, ])

for ticker in tickers:
    try:
        df_vols_cube = load(os.path.join(picklefolder, f'iv_ts_cube_{ticker}.pkl'))
    except:
        df_vols = pd.read_csv(os.path.join(filtered_dir, f'{ticker}.csv'), sep=';')
        df_vols_cube = build_ts_cube(df_vols, maturity_grid, moneyness_grid)
        dump(df_vols_cube, os.path.join(picklefolder, f'iv_ts_cube_{ticker}.pkl'))

    df_vols_cube = df_vols_cube[cols]
    df_vols_cube = df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS.isin(moneyness)]

    df_vols_cube.iloc[:, 1:] **= 2
    model = DynamicSurfaceModels(df_surface=df_vols_cube)
    model.specify_measurement_equation(mean='carr_wu_cyclical_spline', variance='iid',
                                       moneyness_pillars=moneyness_pillars, maturity_pillars=maturity_pillars,
                                       cyclical_knots=cyclical_knots)

    model.fit(cross_sectional=True)

