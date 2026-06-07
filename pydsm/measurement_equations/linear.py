import numpy as np
from numba import jit
from numba.typed import List
from measurement_equations.numerical_jacobian import JacTwoSided
from measurement_equations.carr_wu_standard import carr_wu_iv, map_moments, Jkappa, Jomega, Jnu, Jrho
from measurement_equations.carr_wu_seasonal_fixed import seasonal_effect


@jit(nopython=True, cache=True)
def linear(f, FIXED, betas, p, m):
    x, tau, T, Z = FIXED
    return Z @ f

@jit(nopython=True, cache=True)
def jac_linear(Zf, f, FIXED, betas, p, k):
    x, tau, T, Z = FIXED
    return Z

class Linear:
    def __init__(self, mean: str = 'linear'):
        self.mean = mean
        self.Zf = linear
        self.J = jac_linear
        self.k_fxd = 0
        self.k_free = 0
        self.k_betas_free = 0
        self.k_betas_pos = 0
        # self._test_jac()

    # def _test_params(self):
    #     p = 10
    #     # f = np.array([-.95, -.45, -.35, .333, -0.225]).reshape(self.k, 1)
    #     f = np.array([-.95, -.45, -.35, .333, 0]).reshape(self.k, 1)
    #     x = np.array([-0.4, -0.4, -0.3, -0.3, 0, 0, .3, .3, 0.4, 0.4,]).reshape(p, 1)
    #     tau = np.array([1, 12, 1, 12, 1, 12, 1, 12, 1, 12]).reshape(p, 1) /12
    #     Z = np.ones((p, self.k))
    #     betas = np.array([-10, 2, 3, 4, 5, 6]).reshape(6,1)/10
    #
    #     T = tau + 0.125
    #     FIXED = List()
    #     FIXED.append(x)
    #     FIXED.append(tau)
    #     FIXED.append(T)
    #     FIXED.append(Z)
    #
    #     args = (f, FIXED, betas, p, self.k)
    #     return args
    #
    # def _test_jac(self):
    #     if self.J is not None:
    #         args = self._test_params()
    #         Jtrue = self.J(self.Zf, *args)
    #         Japprox = JacTwoSided(self.Zf, *args)
    #         error = ((Jtrue - Japprox)**2).mean(axis=0)**.5
    #         print(error.round(5))
    #         print('Analytical Jacobian Test Passed:', np.allclose(Jtrue, Japprox, atol=3e-2))
    #         print(Jtrue)
#
# cwsf = Linear()
# cwsf._test_jac()



