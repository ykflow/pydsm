from measurement_equations.depreciated.equations import *
# from measurement_equations.jacobians import *
from scipy.interpolate import interp1d
from measurement_equations.carr_wu_standard import CarrWuStandard
from measurement_equations.carr_wu_prop_fixed import CarrWuPropFixed
from measurement_equations.carr_wu_seasonal_fixed import CarrWuSeasonalFixed
from measurement_equations.carr_wu_prop_seasonal_fixed import CarrWuPropSeasonalFixed
from measurement_equations.carr_wu_disprop_seasonal_fixed import CarrWuDisPropSeasonalFixed
from measurement_equations.carr_wu_seasonal import CarrWuSeasonal
from measurement_equations.carr_wu_prop_seasonal_dyn_fixed import CarrWuPropSeasonalDynamicFixed
from measurement_equations.carr_wu_disprop_seasonal_dyn_fixed import CarrWuDisPropSeasonalDynamicFixed
from measurement_equations.carr_wu_prop_seasonal import CarrWuPropSeasonal
from measurement_equations.carr_wu_disprop_seasonal import CarrWuDisPropSeasonal
from measurement_equations.carr_wu_disprop2_seasonal import CarrWuDisProp2Seasonal
from measurement_equations.carr_wu_disprop2_seasonal_time_space import CarrWuDisProp2SeasonalTimeSpace
from measurement_equations.carr_wu_disprop2_seasonal_knots import CarrWuDisProp2SKnots
from measurement_equations.carr_wu_disprop_seasonal_vol import CarrWuDisPropSeasonalVol
# from measurement_equations.carr_wu_disprop2seasonal_corr import CarrWuDisPropSeasonalCorr
from measurement_equations.carr_wu_disprop2_seasonal_corr import CarrWuDisProp2SeasonalCorr
from measurement_equations.carr_wu_disprop2_seasonal_corr_nu import CarrWuDisProp2SeasonalCorrNu
from measurement_equations.carr_wu_disprop_seasonal_prop_volcorr import CarrWuDisPropSeasonalPropVolCorr
from measurement_equations.carr_wu_seasonal_decay import CarrWuSeasonalDecay
from measurement_equations.carr_wu_seasonal_decay_kappa import CarrWuSeasonalDecayKappa
from measurement_equations.carr_wu_seasonal_decay_omega import CarrWuSeasonalDecayOmega
from measurement_equations.carr_wu_seasonal_decay_rho import CarrWuSeasonalDecayRho
from measurement_equations.carr_wu_seasonal_decay_nu import CarrWuSeasonalDecayNu
from measurement_equations.carr_wu_seasonal_decay_kappa_omega import CarrWuSeasonalDecayKappaOmega
from measurement_equations.carr_wu_seasonal_decay_kappa_rho import CarrWuSeasonalDecayKappaRho
from measurement_equations.carr_wu_seasonal_decay_kappa_nu import CarrWuSeasonalDecayKappaNu
from measurement_equations.carr_wu_seasonal_decay_omega_rho import CarrWuSeasonalDecayOmegaRho
from measurement_equations.carr_wu_seasonal_decay_omega_nu import CarrWuSeasonalDecayOmegaNu
from measurement_equations.carr_wu_seasonal_decay_rho_nu import CarrWuSeasonalDecayRhoNu
from measurement_equations.carr_wu_seasonal_decay_kappa_omega_rho import CarrWuSeasonalDecayKappaOmegaRho
from measurement_equations.carr_wu_seasonal_decay_kappa_omega_nu import CarrWuSeasonalDecayKappaOmegaNu
from measurement_equations.carr_wu_seasonal_decay_kappa_rho_nu import CarrWuSeasonalDecayKappaRhoNu
from measurement_equations.carr_wu_seasonal_decay_omega_rho_nu import CarrWuSeasonalDecayOmegaRhoNu
from measurement_equations.carr_wu_seasonal_decay_kappa_omega_rho_nu import CarrWuSeasonalDecayKappaOmegaRhoNu
from measurement_equations.carr_wu_custom import CarrWuCustom
# from measurement_equations.carr_wu_seasonal_corr import CarrWuSeasonalRho
from measurement_equations.linear import Linear


class SurfaceMeasurementDesignBuilder:
    def __init__(self, moneyness_grid:np.array, maturity_grid: np.array):
        self.moneyness_grid = moneyness_grid
        self.maturity_grid = maturity_grid
        self.dim_p = len(moneyness_grid) * len(maturity_grid)
        self._mean_types = ['carr_wu_standard']
        self._variance_types = ['iid', 'diag', 'mvn', 'mvn-poly']
        self._u_x = None
        self._v_tau = None

    def specify_measurement_equation(self, mean:str='carr_wu_standard', variance:str='iid',
                                     moneyness_pillars:np.array=None, maturity_pillars:np.array=None,
                                     k_factors_linear:int=None, k_ar1_factors:int=0, k_cyclical_knots=None):
        self._mean = mean
        self._variance = variance
        self._moneyness_pillars = moneyness_pillars
        self._maturity_pillars = maturity_pillars
        self._k_factors_linear = k_factors_linear
        self._k_ar1_factors = k_ar1_factors
        self._k_cyclical_knots = k_cyclical_knots

        if self._mean == 'carr_wu_standard':
            self.design = CarrWuStandard()

        elif self._mean == 'carr_wu_prop_fixed':
            self.design = CarrWuPropFixed()

        elif self._mean in ['carr_wu_s1_fixed', 'carr_wu_s2_fixed', 'carr_wu_s3_fixed', 'carr_wu_s4_fixed']:
            self.design = CarrWuSeasonalFixed(self._mean)

        elif self._mean in ['carr_wu_prop_s1_fixed', 'carr_wu_prop_s2_fixed', 'carr_wu_prop_s3_fixed', 'carr_wu_prop_s4_fixed']:
            self.design = CarrWuPropSeasonalFixed(self._mean)

        elif self._mean in ['carr_wu_disprop_s1_fixed', 'carr_wu_disprop_s2_fixed', 'carr_wu_disprop_s3_fixed', 'carr_wu_disprop_s4_fixed']:
            self.design = CarrWuDisPropSeasonalFixed(self._mean)

        elif self._mean in ['carr_wu_s1', 'carr_wu_s2', 'carr_wu_s3', 'carr_wu_s4']:
            self.design = CarrWuSeasonal(self._mean)

        elif self._mean in ['carr_wu_prop_s1_dyn_fixed', 'carr_wu_prop_s2_dyn_fixed', 'carr_wu_prop_s3_dyn_fixed', 'carr_wu_prop_s4_dyn_fixed']:
            self.design = CarrWuPropSeasonalDynamicFixed(self._mean)

        elif self._mean in ['carr_wu_disprop_s1_dyn_fixed', 'carr_wu_disprop_s2_dyn_fixed', 'carr_wu_disprop_s3_dyn_fixed', 'carr_wu_disprop_s4_dyn_fixed']:
            self.design = CarrWuDisPropSeasonalDynamicFixed(self._mean)

        elif self._mean in ['carr_wu_prop_s1', 'carr_wu_prop_s2', 'carr_wu_prop_s3', 'carr_wu_prop_s4']:
            self.design = CarrWuPropSeasonal(self._mean)

        elif self._mean in ['carr_wu_disprop_s1', 'carr_wu_disprop_s2', 'carr_wu_disprop_s3', 'carr_wu_disprop_s4']:
            self.design = CarrWuDisPropSeasonal(self._mean)

        elif self._mean in ['carr_wu_disprop2_s1', 'carr_wu_disprop2_s2', 'carr_wu_disprop2_s3', 'carr_wu_disprop2_s4']:
            self.design = CarrWuDisProp2Seasonal(self._mean)

        elif self._mean in ['carr_wu_disprop2_s1_time_space', 'carr_wu_disprop2_s2_time_space', 'carr_wu_disprop2_s3_time_space', 'carr_wu_disprop2_s4_time_space']:
            self.design = CarrWuDisProp2SeasonalTimeSpace(self._mean)

        elif self._mean in ['carr_wu_disprop2_s1_corr', 'carr_wu_disprop2_s2_corr', 'carr_wu_disprop2_s3_corr', 'carr_wu_disprop2_s4_corr']:
            self.design = CarrWuDisProp2SeasonalCorr(self._mean)

        elif self._mean in ['carr_wu_disprop2_s1_corr_nu', 'carr_wu_disprop2_s2_corr_nu', 'carr_wu_disprop2_s3_corr_nu', 'carr_wu_disprop2_s4_corr_nu']:
            self.design = CarrWuDisProp2SeasonalCorrNu(self._mean)

        elif self._mean in ['carr_wu_disprop_s1_vol', 'carr_wu_disprop_s2_vol', 'carr_wu_disprop_s3_vol', 'carr_wu_disprop_s4_vol']:
            self.design = CarrWuDisPropSeasonalVol(self._mean)

        elif self._mean in ['carr_wu_disprop_s1_prop_volcorr', 'carr_wu_disprop_s2_prop_volcorr', 'carr_wu_disprop_s3_prop_volcorr', 'carr_wu_disprop_s4_prop_volcorr']:
            self.design = CarrWuDisPropSeasonalPropVolCorr(self._mean)

########################### FOR CROSS  SECTIONAL ANALYSIS
        elif self._mean in ['carr_wu_s1_decay', 'carr_wu_s2_decay', 'carr_wu_s3_decay', 'carr_wu_s4_decay']:
            self.design = CarrWuSeasonalDecay(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa', 'carr_wu_s2_decay_kappa', 'carr_wu_s3_decay_kappa', 'carr_wu_s4_decay_kappa']:
            self.design = CarrWuSeasonalDecayKappa(self._mean)

        elif self._mean in ['carr_wu_s1_decay_omega', 'carr_wu_s2_decay_omega', 'carr_wu_s3_decay_omega', 'carr_wu_s4_decay_omega']:
            self.design = CarrWuSeasonalDecayOmega(self._mean)

        elif self._mean in ['carr_wu_s1_decay_rho', 'carr_wu_s2_decay_rho', 'carr_wu_s3_decay_rho', 'carr_wu_s4_decay_rho']:
            self.design = CarrWuSeasonalDecayRho(self._mean)

        elif self._mean in ['carr_wu_s1_decay_nu', 'carr_wu_s2_decay_nu', 'carr_wu_s3_decay_nu', 'carr_wu_s4_decay_nu']:
            self.design = CarrWuSeasonalDecayNu(self._mean)

        elif self._mean == 'carr_wu_disprop2_sknots':
            self.design = CarrWuDisProp2SKnots(self._mean)

        elif self._mean == 'custom':
            self.design = CarrWuCustom(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa_omega', 'carr_wu_s2_decay_kappa_omega', 'carr_wu_s3_decay_kappa_omega', 'carr_wu_s4_decay_kappa_omega']:
            self.design = CarrWuSeasonalDecayKappaOmega(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa_rho', 'carr_wu_s2_decay_kappa_rho', 'carr_wu_s3_decay_kappa_rho', 'carr_wu_s4_decay_kappa_rho']:
            self.design = CarrWuSeasonalDecayKappaRho(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa_nu', 'carr_wu_s2_decay_kappa_nu', 'carr_wu_s3_decay_kappa_nu', 'carr_wu_s4_decay_kappa_nu']:
            self.design = CarrWuSeasonalDecayKappaNu(self._mean)

        elif self._mean in ['carr_wu_s1_decay_omega_rho', 'carr_wu_s2_decay_omega_rho', 'carr_wu_s3_decay_omega_rho', 'carr_wu_s4_decay_omega_rho']:
            self.design = CarrWuSeasonalDecayOmegaRho(self._mean)

        elif self._mean in ['carr_wu_s1_decay_omega_nu', 'carr_wu_s2_decay_omega_nu', 'carr_wu_s3_decay_omega_nu', 'carr_wu_s4_decay_omega_nu']:
            self.design = CarrWuSeasonalDecayOmegaNu(self._mean)

        elif self._mean in ['carr_wu_s1_decay_rho_nu', 'carr_wu_s2_decay_rho_nu', 'carr_wu_s3_decay_rho_nu', 'carr_wu_s4_decay_rho_nu']:
            self.design = CarrWuSeasonalDecayRhoNu(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa_omega_rho', 'carr_wu_s2_decay_kappa_omega_rho', 'carr_wu_s3_decay_kappa_omega_rho', 'carr_wu_s4_decay_kappa_omega_rho']:
            self.design = CarrWuSeasonalDecayKappaOmegaRho(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa_omega_nu', 'carr_wu_s2_decay_kappa_omega_nu', 'carr_wu_s3_decay_kappa_omega_nu', 'carr_wu_s4_decay_kappa_omega_nu']:
            self.design = CarrWuSeasonalDecayKappaOmegaNu(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa_rho_nu', 'carr_wu_s2_decay_kappa_rho_nu', 'carr_wu_s3_decay_kappa_rho_nu', 'carr_wu_s4_decay_kappa_rho_nu']:
            self.design = CarrWuSeasonalDecayKappaRhoNu(self._mean)

        elif self._mean in ['carr_wu_s1_decay_omega_rho_nu', 'carr_wu_s2_decay_omega_rho_nu', 'carr_wu_s3_decay_omega_rho_nu', 'carr_wu_s4_decay_omega_rho_nu']:
            self.design = CarrWuSeasonalDecayOmegaRhoNu(self._mean)

        elif self._mean in ['carr_wu_s1_decay_kappa_omega_rho_nu', 'carr_wu_s2_decay_kappa_omega_rho_nu', 'carr_wu_s3_decay_kappa_omega_rho_nu', 'carr_wu_s4_decay_kappa_omega_rho_nu']:
            self.design = CarrWuSeasonalDecayKappaOmegaRhoNu(self._mean)



        elif self._mean == 'linear':
            self.design = Linear()

        else:
            raise ValueError

        self._Zf = self.design.Zf
        self._k_free = self.design.k_free + self._k_ar1_factors
        self._k_fixed = self.design.k_fxd - self._k_ar1_factors
        self._k_betas_free = self.design.k_betas_free
        self._k_betas_pos = self.design.k_betas_pos
        self._jacobian = self.design.J

        if self._mean in ['linear', 'exponential']:
            self._k_fixed = self._k_factors_linear - self._k_ar1_factors
            self._k_free =  self._k_ar1_factors

        if self._mean == 'carr_wu_disprop2_sknots':
            self._k_fixed += self._k_cyclical_knots

        # self._Zf = carr_wu_prop2_s1_fixed
        # self._k_free = 0
        # self._k_fixed = 4
        # self._k_betas_free = 2 + 2
        # self._k_betas_pos = 0
        # self._jacobian = None

        self._k_factors = self._k_free + self._k_fixed
        self._p_x = 0
        if self._variance == 'iid':
            self._p_pillars = 1

        elif self._variance == 'diag':
            self._p_pillars = self.dim_p

        elif self._variance == 'mvn':
            self._p_x = self.moneyness_grid.shape[0] -1
            self._p_tau = self.maturity_grid.shape[0]
            self._p_pillars = self._p_x + self._p_tau
            self._idx_x = self._moneyness_pillars != 0

        elif self._variance == 'mvn-spline':
            self._p_x = moneyness_pillars.shape[0] -1
            self._p_tau = maturity_pillars.shape[0]
            self._p_pillars = self._p_x + self._p_tau
            self._idx_x = self._moneyness_pillars != 0

        elif self._variance == 'mvn-poly':
            self._p_pillars = 2 + 3

        else:
            raise ValueError

    def _build_covH(self, H_variances):
        if self._variance == 'iid':
            return np.eye(self.dim_p) * H_variances
        elif self._variance == 'diag':
            return np.diag(H_variances.flatten())
        elif self._variance == 'mvn':
            H_x = np.ones((self._p_x+1, 1))
            H_x[self._idx_x] = H_variances[:self._p_x]
            H_tau = H_variances[self._p_x:]
            U = np.diag(H_x.flatten())
            V = np.diag(H_tau.flatten())
            return np.kron(V, U)
        elif self._variance == 'mvn-poly':
            x, tau = self.moneyness_grid.flatten(), self.maturity_grid.flatten()
            u1, u2, v0, v1, v2 = np.log(H_variances).flatten()
            U = np.diag(np.exp(u1*x + u2*x**2))
            V = np.diag(np.exp(v0 + v1*tau + v2*tau**2))
            return np.kron(V, U)

        elif self._variance == 'mvn-spline':
            x, tau = self.moneyness_grid.flatten(), self.maturity_grid.flatten()
            H_x = np.ones((self._p_x+1, 1))
            H_x[self._idx_x] = H_variances[:self._p_x]
            H_tau = H_variances[self._p_x:]

            self._u_x = interp1d(self._moneyness_pillars.flatten(), H_x.flatten(),
                                        fill_value='extrapolate')
            self._v_tau = interp1d(self._maturity_pillars.flatten(), H_tau.flatten(),
                                      fill_value='extrapolate')

            U = np.diag(self._u_x(x))
            V = np.diag(self._v_tau(tau))
            return np.kron(V, U)

        elif self._variance == 'mvn-spline-restricted':
            x, tau = self.moneyness_grid.flatten(), self.maturity_grid.flatten()
            H_x = np.ones((self._p_x+1, 1))
            H_x[self._idx_x] = H_variances[:self._p_x]

            H_tau = H_variances[self._p_x:].flatten()
            H_tau = np.concatenate((H_tau[0],H_tau))
            tau = np.concatenate((np.array([0]), self._maturity_pillars.flatten(),))

            self._u_x = interp1d(self._moneyness_pillars.flatten(), H_x.flatten(),
                                        fill_value='extrapolate')
            self._v_tau = interp1d(tau, H_tau,
                                      fill_value='extrapolate')

            U = np.diag(self._u_x(x))
            V = np.diag(self._v_tau(tau))
            return np.kron(V, U)


        else:
            raise ValueError




    # def overwrite_maturities(self, tau: np.array, timeline):
    #     self.tau = tau
    #     self.timeline = timeline
    #     n,p = self.tau.shape
    #     self._t = np.array([self.day_of_year(t) for t in self.timeline]).reshape(n,1)
    #
    # def build_design_matrix(self, overwrite_tau: np.array=None, overwrite_timeline: np.array=None):
    #     tau = self.tau if overwrite_tau is None else overwrite_tau
    #     timeline = self.timeline if overwrite_timeline is None else overwrite_timeline
    #     n, p = tau.shape
    #     t = self._t if overwrite_timeline is None else np.array([self.day_of_year(t) for t in timeline]).reshape(n,1)
    #
    #
    @property
    def Zf(self):
        return self._Zf
