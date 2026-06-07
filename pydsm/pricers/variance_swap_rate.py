import pandas as pd
import numpy as np
from scipy.interpolate import  interp1d
from pricers.black_futures_options import BlackFuturesOptionsPricer

class VarianceSwapPricer:
    def __init__(self, variance_surfaces: pd.DataFrame, forwards: pd.DataFrame, rates: pd.DataFrame):
        self.variance_surfaces = variance_surfaces
        self.forwards = forwards
        self.rates = rates
        self._time_line = self.variance_surfaces.index.unique()
        self.x, self.tau = self.variance_surfaces.CLOSEST_MONEYNESS.unique(), self.variance_surfaces.columns[1:].to_numpy()
        self.m, self.n = len(self.x), len(self.tau)
        self.X = np.ones((self.m, self.n)) * self.x.reshape(self.m,1)
        self.TAU = np.ones((self.m, self.n)) * self.tau.reshape(self.n, 1).T
        self.interpolator = interp1d
        self.pricer = BlackFuturesOptionsPricer

    def surface_interpolator(self):
        variances = self.variance_surfaces.copy().reset_index().set_index(['DATE', 'CLOSEST_MONEYNESS'])
        scale = np.kron(np.ones((len(self._time_line), 1)), self.TAU)
        total_variances = variances * scale
        total_variances = total_variances.astype(float).interpolate(method='linear', axis=1)
        variances = total_variances / scale
        variances = variances.T.bfill().ffill().T
        self.variance_surfaces = variances.reset_index().set_index('DATE')


    def _trapzoid_integration(self, fx, deltax):
        return deltax * ((fx[0] + fx[-1]) / 2 + fx[1:-1].sum())

    def compute_variance_swap_rates(self):
        variance_swap_rates = pd.DataFrame(index=self._time_line, columns=self.tau)
        for t in self._time_line:
            F = self.forwards.loc[t, self.tau].values.reshape(self.n,1).T
            R = self.rates.loc[t, self.tau].values.flatten()
            I2 = self.variance_surfaces.loc[t, self.tau].values
            K = np.exp(self.X) * F

            i = 0
            for tau in self.tau:
                K_i = K[:, i]
                F_i = F.flatten()[i]
                I2_i = I2[:, i]
                r_i = R[i]

                K_min, K_max = K_i.min(), K_i.max()
                K_grid = np.linspace(K_min, K_max, 2000, endpoint=True)
                deltaK = K_grid[1] - K_grid[0]
                I2_grid = self.interpolator(K_i, I2_i, kind='cubic')(K_grid)
                loc_otm_call = K_grid > F_i
                loc_otm_put = K_grid <= F_i
                pricer_call = self.pricer(F_i, K_grid[loc_otm_call], tau, I2_grid[loc_otm_call], r_i)
                pricer_put = self.pricer(F_i, K_grid[loc_otm_put], tau, I2_grid[loc_otm_put], r_i)
                C_grid_otm = pricer_call.call_price()
                P_grid_otm = pricer_put.put_price()

                vsr1 = (1 / tau) * self._trapzoid_integration(fx=C_grid_otm / (K_grid[loc_otm_call] ** 2),  deltax=deltaK)
                vsr2 = (1 / tau) * self._trapzoid_integration(fx=P_grid_otm / (K_grid[loc_otm_put] ** 2),  deltax=deltaK)
                variance_swap_rates.loc[t, tau] = vsr1 + vsr2

        self.variance_swap_rates = variance_swap_rates




#
#
#
#
# #### CONFIG
# import os
# from pathlib import Path
# from config_utils.config_utils_main import import_ts_ivs
# from calculator.forward_curve_calculator import ForwardCurveCalculator
# from calculator.yield_curve_calculator import ZeroRatesCalculator
# import matplotlib.pyplot as plt
# codefolder = Path(os.path.dirname(os.path.realpath(__file__)))
# basefolder = codefolder.parent
# datafolder = os.path.join(basefolder, 'data_storage', )
# plotfolder = os.path.join(basefolder, 'plot_folder', )
# picklefolder = os.path.join(basefolder, 'pickle_folder', )
# tablefolder = os.path.join(basefolder, 'table_folder', )
# HUNDRED = 100
# days = 365
#
# ### CONFIG
# ticker = 'NG Pen Futures 25k ICE Lots_Henry_HH'
# x_grid = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
# x_knots = np.array([-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3])
# tau_grid = np.linspace(0, 50/24, 51, endpoint=True).round(4)[1:]
# tau_knots = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12])
# cyclical_knots = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])
# df_variances = import_ts_ivs(x_grid, tau_grid, ticker)
# df_variances.iloc[:, 1:] **= 2
# timeline = df_variances.index.unique()
#
# fcc = ForwardCurveCalculator(os.path.join(datafolder, 'comdty_forwards'))
# zrc = ZeroRatesCalculator(os.path.join(datafolder, 'rates'))
#
# df_forwards = fcc.interpolate(tau_grid, timeline)
# df_rates = zrc.interpolate(tau_grid, timeline)
#
# vsp = VarianceSwapPricer(df_variances, df_forwards, df_rates)
# vsp.surface_interpolator()
# vsp.compute_variance_swap_rates()
# vsp.variance_swap_rates[[0.0833, 0.25, 0.5, 1, 1.5, 2]].plot(figsize=(20,10)), plt.tight_layout(), plt.show()


