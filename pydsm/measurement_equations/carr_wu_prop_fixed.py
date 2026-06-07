import numpy as np
from numba import jit
from numba.typed import List
from measurement_equations.numerical_jacobian import JacTwoSided
from measurement_equations.carr_wu_standard import carr_wu_iv, map_moments, Jkappa, Jomega, Jnu, Jrho


@jit(nopython=True, cache=True)
def carr_wu_prop_fixed(f, FIXED, betas, p, m):
    x, tau, T, Z = FIXED
    kappa1, omega1_star, nu_star, rho_star = f.flatten()
    eta_tilde = betas.flatten()[0]
    omega1, nu, rho, gamma = map_moments(omega1_star, nu_star, rho_star)
    eta = np.exp(eta_tilde)
    exp_eta_tau = np.exp(-eta*tau)
    kappa = kappa1 * exp_eta_tau
    omega = omega1 * exp_eta_tau
    return carr_wu_iv(kappa, omega, nu, rho, gamma, x, tau)


@jit(nopython=True, cache=True)
def jac_carr_wu_prop_fixed(Zf, f, FIXED, betas, p, k):
    x, tau, T, Z = FIXED
    kappa1, omega1_star, nu_star, rho_star = f.flatten()
    eta_tilde = betas.flatten()[0]
    omega1, nu, rho, gamma = map_moments(omega1_star, nu_star, rho_star)
    eta = np.exp(eta_tilde)
    exp_eta_tau = np.exp(-eta*tau)
    kappa = kappa1 * exp_eta_tau
    omega = omega1 * exp_eta_tau

    tmp = 1 - 2 * kappa * tau - gamma * tau
    a = -2 * tmp / (omega ** 2 * tau ** 2)
    b = - rho * nu / omega
    c = ((1 - rho ** 2) * (nu ** 2) / (omega ** 2)) + tmp ** 2 / (omega ** 4 * tau ** 2)
    denom = tau * ((x - b) ** 2 + c)**.5

    z_kappa = Jkappa(omega, b, c, tmp, x, tau) #dmu/dkappa
    z_omega = Jomega(omega, nu, rho, a, b, tmp, denom, x, tau) #dmu/domega
    z_nu = Jnu(omega, nu, rho, b, tmp, denom, x, tau)
    z_rho = Jrho(omega, nu, tmp, denom, x, tau)

    z1 = z_kappa * exp_eta_tau  #dmu/dk1
    z2 = z_omega * omega #* exp_eta_tau#dmu/domega_star1
    z3 = z_nu * nu
    z4 = z_rho * (4 * np.exp(2 * rho_star) / (np.exp(2 * rho_star) + 1) ** 2)
    z5 = (z_kappa + z_omega) * tau * exp_eta_tau

    return np.concatenate((z1, z2, z3, z4, z5), axis=1)


class CarrWuPropFixed:
    def __init__(self):
        self.Zf = carr_wu_prop_fixed
        # self.J = jac_carr_wu_prop
        self.J = None
        self.k_fxd = 4
        self.k_free = 0
        self.k_betas_free = 1
        self.k_betas_pos = 0
        self._test_jac()

    def _test_params(self):
        p = 10
        # f = np.array([-.95, -.45, -.35, .333, -0.225]).reshape(self.k, 1)
        f = np.array([-.95, -.45, -.35, .333, 0]).reshape(self.k_fxd, 1)
        x = np.array([-0.4, -0.4, -0.3, -0.3, 0, 0, .3, .3, 0.4, 0.4,]).reshape(p, 1)
        tau = np.array([1, 12, 1, 12, 1, 12, 1, 12, 1, 12]).reshape(p, 1) /12
        Z = np.ones((p, self.k_fxd))
        betas = np.array([-10, 2, 3, 4, 5, 6]).reshape(6,1)/10

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

# cwsf = CarrWuPropFixed()
# cwsf._test_jac()



