import os
import pandas as pd
import numpy as np


class ForwardCurveCalculator:
    def __init__(self, curves_folder: os.path, ticker:str='HH'):
        self.forwards = pd.read_csv(os.path.join(curves_folder, 'commodity_futures_feb2007-aug2024.csv'))
        self.ticker = ticker
        self.forwards = self.forwards[self.forwards.KEY == self.ticker].drop(columns=['KEY'])
        self.forwards['DATE'] = pd.to_datetime(self.forwards['DATE'], format='%m/%d/%Y')
        self.forwards.set_index('DATE', inplace=True)

        self.interpolator = np.interp

    def interpolate(self, tau, index: pd.DatetimeIndex):
        df = pd.DataFrame(index=index, columns=tau)
        for t in index:
            try:
                tmp = self.forwards.loc[t]
                x, y = tmp.X.values.flatten(), tmp.VALUE.values.flatten()
                df.loc[t] = self.interpolator(tau, x, y)
            except:
                pass

        df = df.astype(float).ffill().bfill()
        return df

# folder = 'C:/Users/PI26UT/OneDrive - ING/Desktop/Documents/PhD/Article IV/pyDynamicSurfaceModels/data_storage/comdty_forwards'
# fcc = ForwardCurveCalculator(folder)





