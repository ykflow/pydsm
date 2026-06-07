import numpy as np
from numba import jit
from numba.typed import List
from measurement_equations.numerical_jacobian import JacTwoSided


@jit(nopython=True, cache=True)
def carr_wu_iv(kappa, omega, nu, rho, gamma, x, tau):
    tmp = 1 - 2 * kappa * tau - gamma * tau
    a = -2 * tmp / (omega ** 2 * tau ** 2)
    b = - rho * nu / omega
    c = ((1 - rho ** 2) * (nu ** 2) / (omega ** 2)) + tmp ** 2 / (omega ** 4 * tau ** 2)
    I2 = a + (2 / tau) * ((x - b) ** 2 + c)**.5
    return I2

@jit(nopython=True, cache=True)
def map_moments(omega_star, nu_star, rho_star):
    omega = np.exp(omega_star)
    nu = np.exp(nu_star)
    rho = (np.exp(rho_star) - np.exp(-rho_star)) / (np.exp(rho_star) + np.exp(-rho_star))
    gamma = rho * omega * nu
    return omega, nu, rho, gamma

@jit(nopython=True, cache=True)
def carr_wu_standard(f, FIXED, betas, p, m):
    x, tau, T, Z = FIXED
    kappa, omega_star, nu_star, rho_star = f.flatten()
    omega, nu, rho, gamma = map_moments(omega_star, nu_star, rho_star)
    return carr_wu_iv(kappa, omega, nu, rho, gamma, x, tau)

@jit(nopython=True, cache=True)
def Jkappa(omega, b, c, tmp, x, tau):
    z1 = 4 / (omega ** 2 * tau) - 4 * tmp / (omega ** 4 * tau ** 2 * ((x - b) ** 2 + c) ** .5)
    return z1

@jit(nopython=True, cache=True)
def Jomega(omega, nu, rho, a, b, tmp, denom, x, tau):
    z2 = -2 * (1 / omega * (a + b / tau)
         + (1 * ((1 - rho ** 2) * (nu ** 2 / omega ** 3) + 2 * tmp ** 2 / (omega ** 5 * tau ** 2))
            - 1 * (b * ((x - b) / omega + tmp / (omega ** 3 * tau)))
            ) / denom
              )
    return z2

@jit(nopython=True, cache=True)
def Jnu(omega, nu, rho, b, tmp, denom, x, tau):
    z3 = 2 * (rho / (omega * tau) + (
                rho * (x - b) / omega + nu * (1 - rho ** 2) / omega ** 2 - rho * tmp / (omega ** 3 * tau)) / denom)
    return z3

@jit(nopython=True, cache=True)
def Jrho(omega, nu, tmp, denom, x, tau):
    z4 = 2 * (nu / (omega * tau) + nu * (x / omega - tmp / (omega ** 3 * tau)) / denom)
    return z4

@jit(nopython=True, cache=True)
def jac_carr_wu_standard(Zf, f, FIXED, betas, p, k):
    x, tau, T, Z = FIXED
    kappa, omega_star, nu_star, rho_star = f.flatten()
    omega, nu, rho, gamma = map_moments(omega_star, nu_star, rho_star)

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
    z2 = z_omega * omega
    z3 = z_nu * nu
    z4 = z_rho * (4 * np.exp(2 * rho_star) / (np.exp(2 * rho_star) + 1) ** 2)

    return np.concatenate((z1, z2, z3, z4), axis=1)


class CarrWuStandard:
    def __init__(self):
        self.Zf = carr_wu_standard
        # self.J = jac_carr_wu_standard
        self.J = None
        self.k_fxd = 4
        self.k_free = 0
        self.k_betas_free = 0
        self.k_betas_pos = 0

        self._test_jac()

    def _test_params(self):
        p = 10
        f = np.array([-.95, -.45, -.35, .333]).reshape(self.k_fxd, 1)
        x = np.array([-0.4, -0.4, -0.3, -0.3, 0, 0, .3, .3, 0.4, 0.4]).reshape(p, 1)
        tau = np.array([1, 12, 1, 12, 1, 12, 1, 12, 1, 12]).reshape(p, 1) /12
        Z = np.ones((p, self.k_fxd))
        betas = np.array([])

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


# cws = CarrWuStandard()



