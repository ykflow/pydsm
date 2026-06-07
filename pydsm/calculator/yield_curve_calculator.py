import os
import pandas as pd
import numpy as np


class ZeroRatesCalculator:
    def __init__(self, rates_folder:os.path):
        self.rates = pd.read_csv(os.path.join(rates_folder, 'treasury_securities.csv'))
        self.rates['DATE'] = pd.to_datetime(self.rates['DATE'])
        self.rates.set_index('DATE', inplace=True)
        self.ttm = np.array([1/12, 3/12, 6/12, 1, 2]).round(3)
        self.ttm_cols = ['1M', '3M', '6M', '1Y', '2Y']
        self.rates.columns = ['1Y', '2Y', '3M', '1M', '6M']
        self.rates = self.rates[self.ttm_cols].ffill().bfill() / 100
        self.interpolator = np.interp

    def interpolate(self, tau, index:pd.DatetimeIndex):
        df = pd.DataFrame(index=index, columns=tau)
        for t in index:
            try:
                x, y = self.ttm, self.rates.loc[t].values.flatten()
                df.loc[t] = self.interpolator(tau, x, y)
            except:
                pass

        df = df.astype(float).ffill().bfill()
        return df


# rates_folder = 'C:/Users/PI26UT/OneDrive - ING/Desktop/Documents/PhD/Article IV/pyDynamicSurfaceModels/data_storage/rates'
# zrc = ZeroRatesCalculator(rates_folder)





