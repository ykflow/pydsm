import numpy as np
import pandas as pd
from joblib import load, dump
import os
from pathlib import Path
from data_utils.build_ts_cubes import build_ts_cube

def import_ts_ivs(x:np.array, tau:np.array, ticker:str ='NG Pen Futures 25k ICE Lots_Henry_HH'):
    #### CONFIG
    codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
    basefolder = codefolder.parent
    picklefolder = os.path.join(basefolder, 'pickle_folder', )
    moneyness_grid = np.linspace(-.5, .5, 41, endpoint=True).round(4)  # 0.025 step size
    maturity_grid = np.linspace(0, 50/24, 51, endpoint=True) # bi-weekly step size
    maturity_grid = np.concatenate((np.array([1/48]), maturity_grid)).round(4)
    maturity_grid.sort()
    filtered_dir = 'C:/Users/PI26UT/Documents/Data/ComdtyVols/Filtered Data'
    cols = ['CLOSEST_MONEYNESS'] + tau.tolist()

    try:
        df_vols_cube = load(os.path.join(picklefolder, f'iv_ts_cube_{ticker}.pkl'))
    except:
        df_vols = pd.read_csv(os.path.join(filtered_dir, f'{ticker}.csv'), sep=';')
        df_vols_cube = build_ts_cube(df_vols, maturity_grid, moneyness_grid)
        dump(df_vols_cube, os.path.join(picklefolder, f'iv_ts_cube_{ticker}.pkl'))

    df_vols_cube = df_vols_cube[cols]
    df_vols_cube = df_vols_cube[df_vols_cube.CLOSEST_MONEYNESS.isin(x.tolist())]

    return df_vols_cube
