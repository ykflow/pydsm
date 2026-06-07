import numpy as np
import pandas as pd


def build_ts_cube(df_vols: pd.DataFrame, maturity_grid, grid_moneyness: np.array):
    HUNDRED = 100
    df_vols.DATE = pd.to_datetime(df_vols.DATE, format='%Y%m%d')
    min_x, max_x = grid_moneyness.min(), grid_moneyness.max()

    df_vols = df_vols[(df_vols.EXPIRY_YEARS > 0) & (df_vols.EXPIRY_YEARS <= maturity_grid.max())]
    df_vols = df_vols[(df_vols.MONEYNESS >= min_x) & (df_vols.MONEYNESS <= max_x)]

    find_closest_moneyness = lambda x: min(grid_moneyness, key=lambda y: abs(y - x))
    find_closest_ttm = lambda x: min(maturity_grid, key=lambda y: abs(y - x))
    df_vols['CLOSEST_MONEYNESS'] = df_vols.MONEYNESS.apply(find_closest_moneyness)
    df_vols['CLOSEST_EXPIRY'] = df_vols.EXPIRY_YEARS.apply(find_closest_ttm)

    df_ts_cube = df_vols.groupby(by=['DATE', 'CLOSEST_EXPIRY', 'CLOSEST_MONEYNESS'])['CALL_VOL'].mean().reset_index()
    df_ts_cube = df_ts_cube.pivot_table(index=['DATE', 'CLOSEST_MONEYNESS'], columns='CLOSEST_EXPIRY', values='CALL_VOL')

    df_ts_cube /= HUNDRED
    df_ts_cube = df_ts_cube.reset_index()

    dates = df_ts_cube.DATE.unique()
    timeline = np.repeat(dates, len(grid_moneyness))
    moneyness = grid_moneyness.tolist()* len(dates)

    df_ts_cube_full = pd.DataFrame(timeline, columns=['DATE'])
    df_ts_cube_full['CLOSEST_MONEYNESS'] = moneyness
    df_ts_cube_full = pd.merge(left=df_ts_cube_full, right=df_ts_cube, on=['DATE', 'CLOSEST_MONEYNESS'], how='left')
    df_ts_cube_full.sort_values(by=['DATE', 'CLOSEST_MONEYNESS'], inplace=True)

    df_ts_cube_full.set_index('DATE', inplace=True)
    try:
        df_ts_cube_full.drop(columns=[0.], inplace=True)
    except:
        pass

    return df_ts_cube_full


