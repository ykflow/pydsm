import numpy as np
from numba import jit
from numba.typed import List
from measurement_equations.numerical_jacobian import JacTwoSided
from measurement_equations.carr_wu_standard import carr_wu_iv, map_moments, Jkappa, Jomega, Jnu, Jrho
from measurement_equations.carr_wu_seasonal_fixed import seasonal_effect


@jit(nopython=True, cache=True)
def carr_wu_seasonal_decay_nu(f, FIXED, betas, p, m):
    x, tau, T, Z = FIXED
    kappa1, omega_star, nu1_star, rho_star, eta1_star, eta2_star = f.flatten()[:6]
    deltas = f[6:]
    omega, nu1, rho, gamma = map_moments(omega_star, nu1_star, rho_star)
    S = seasonal_effect(deltas, T)
    kappa = kappa1 + S * np.exp(eta2_star * tau)
    nu = nu1 * np.exp(eta1_star * tau)
    return carr_wu_iv(kappa, omega, nu, rho, gamma, x, tau)


@jit(nopython=True, cache=True)
def jac_carr_wu_seasonal_fxd(Zf, f, FIXED, betas, p, k):
    x, tau, T, Z = FIXED
    kappa1, omega_star, nu_star, rho_star = f.flatten()
    omega, nu, rho, gamma = map_moments(omega_star, nu_star, rho_star)
    S = seasonal_effect(betas, T)
    kappa = kappa1 + S

    tmp = 1 - 2 * kappa * tau - gamma * tau
    a = -2 * tmp / (omega ** 2 * tau ** 2)
    b = - rho * nu / omega
    c = ((1 - rho ** 2) * (nu ** 2) / (omega ** 2)) + tmp ** 2 / (omega ** 4 * tau ** 2)
    denom = tau * ((x - b) ** 2 + c)**.5

    z_kappa = Jkappa(omega, b, c, tmp, x, tau)
    z_omega = Jomega(omega, nu, rho, a, b, tmp, denom, x, tau)
    z_nu = Jnu(omega, nu, rho, b, tmp, denom, x, tau)
    z_rho = Jrho(omega, nu, tmp, denom, x, tau)

    z1 = z_kappa
    z2 = z_omega * omega_star * omega
    z3 = z_nu * nu_star * nu
    z4 = z_rho * (4 * np.exp(2 * rho_star) / (np.exp(2 * rho_star) + 1) ** 2)

    return np.concatenate((z1, z2, z3, z4), axis=1)


class CarrWuSeasonalDecayNu:
    def __init__(self, mean:str = 'carr_wu_s1_decay_nu'):
        self.mean = mean
        self.Zf = carr_wu_seasonal_decay_nu
        # self.J = jac_carr_wu_seasonal_fxd
        self.J = None
        self._dict_means = dict({'carr_wu_s1_decay_nu':2, 'carr_wu_s2_decay_nu':4,
                                 'carr_wu_s3_decay_nu':6, 'carr_wu_s4_decay_nu':8})
        self.k_fxd = 6 + self._dict_means[self.mean]
        self.k_free = 0
        self.k_betas_free = 0
        self.k_betas_pos = 0
        self._test_jac()

    def _test_params(self):
        p = 5
        f = np.array([-3, -.5, -.5, .2]).reshape(self.k_fxd, 1)
        x = np.array([-0.4, -0.3, 0, .5, 1]).reshape(p, 1)
        tau = np.array([1 / 12, 1 / 2, 1 / 12, 1, 1 / 12]).reshape(p, 1)
        Z = np.ones((p, self.k_fxd))
        betas = np.array([2, 4, 6, 8, 10, 12, 7]).reshape(7,1)/20

        T = tau + 0.125
        FIXED = List()
        FIXED.append(x)
        FIXED.append(tau)
        FIXED.append(T)
        FIXED.append(Z)

        args = (f, FIXED, betas, p, self.k_fxd)
        return args

    def _test_jac(self):
        if self.J is not None:
            args = self._test_params()
            Jtrue = self.J(self.Zf, *args)
            Japprox = JacTwoSided(self.Zf, *args)
            error = ((Jtrue - Japprox)**2).mean(axis=0)**.5
            print(error.round(5))
            print('Analytical Jacobian Test Passed:', np.allclose(Jtrue, Japprox, atol=3e-2))
            print(Jtrue)


# cwsf = CarrWuSeasonalFixed()
# # cwsf._test_jac()



