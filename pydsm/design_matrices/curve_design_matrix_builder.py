import numpy as np
from design_matrices.nelson_siegel import NelsonSiegelDesignMatrix
from design_matrices.seasonality import SeasonalityDesignMatrix
from design_matrices.stochastic_seasonality import StochasticSeasonalityDesignMatrix
from datetime import datetime


class CurveDesignMatrixBuilder:
    def __init__(self, tau, timeline):
        self.tau = tau
        n,p = self.tau.shape
        self.timeline = timeline
        self._days = 365
        self._t = np.array([self.day_of_year(t) for t in self.timeline]).reshape(n,1)

    def day_of_year(self, t: datetime) ->np.array:
        return ((t - datetime(t.year, 1, 1)).days + 1) / self._days

    def specify_design_matrix(self, add_fixed_time_seasons=False, add_fixed_curve_seasons=False,
                              add_tvp_time_seasons=False, add_tvp_curve_seasons=False,
                              k_fixed_time_seasons: int = 0, k_fixed_curve_seasons: int = 0,
                              k_tvp_time_seasons: int = 0, k_tvp_curve_seasons: int = 0):

        self._add_fixed_time_seasons = add_fixed_time_seasons
        self._add_fixed_curve_seasons = add_fixed_curve_seasons
        self._add_tvp_time_seasons = add_tvp_time_seasons
        self._add_tvp_curve_seasons = add_tvp_curve_seasons

        self._k_fixed_time_seasons = k_fixed_time_seasons
        self._k_fixed_curve_seasons = k_fixed_curve_seasons
        self._k_tvp_time_seasons = k_tvp_time_seasons
        self._k_tvp_curve_seasons = k_tvp_curve_seasons

        self._m = 3 + 2*(self._k_tvp_time_seasons + self._k_tvp_curve_seasons)
        self._m_free = 3
        self._m_fixed = 2*(self._k_tvp_time_seasons + self._k_tvp_curve_seasons)
        self._k_betas = 2*(self._k_fixed_time_seasons + self._k_fixed_curve_seasons)
        self._k_lmbdas = 1

    def overwrite_maturities(self, tau: np.array, timeline):
        self.tau = tau
        self.timeline = timeline
        n,p = self.tau.shape
        self._t = np.array([self.day_of_year(t) for t in self.timeline]).reshape(n,1)

    def build_design_matrix(self, overwrite_tau: np.array=None, overwrite_timeline: np.array=None,
                            lmbdas=np.array([None]), betas=np.array([None])):
        self._lmbdas = lmbdas.flatten()
        self._betas = betas.flatten()

        tau = self.tau if overwrite_tau is None else overwrite_tau
        timeline = self.timeline if overwrite_timeline is None else overwrite_timeline
        n, p = tau.shape
        t = self._t if overwrite_timeline is None else np.array([self.day_of_year(t) for t in timeline]).reshape(n,1)

        Z = np.ones((n,p,1), dtype=np.float64)
        builder = NelsonSiegelDesignMatrix(self._lmbdas, self.tau)
        Z = np.concatenate((Z, builder.make_design_matrix()), axis=2)

        Xbeta = np.zeros((n, p, 1)) ###ZERO INTERCEPTS
        k_betas_time = 2*self._k_fixed_time_seasons
        if self._add_fixed_time_seasons:
            builder = SeasonalityDesignMatrix(self._k_fixed_time_seasons, self._betas[:k_betas_time], t * np.ones((1,p)))
            Xbeta += builder.make_design_matrix()
        if self._add_fixed_curve_seasons:
            builder = SeasonalityDesignMatrix(self._k_fixed_curve_seasons, self._betas[k_betas_time:], t+tau)
            Xbeta += builder.make_design_matrix()

        if self._add_tvp_time_seasons:
            builder = StochasticSeasonalityDesignMatrix(self._k_tvp_time_seasons, t * np.ones((1,p)))
            Z = np.concatenate((Z, builder.make_design_matrix()), axis=2)
        if self._add_tvp_curve_seasons:
            builder = StochasticSeasonalityDesignMatrix(self._k_tvp_curve_seasons, t + tau)
            Z = np.concatenate((Z, builder.make_design_matrix()), axis=2)

        Z[np.isnan(Z)] = 0.  ## SET ALL ROW WITH NANS TO ZERO
        return Z, Xbeta

