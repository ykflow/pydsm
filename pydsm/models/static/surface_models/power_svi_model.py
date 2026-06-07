import numpy as np
from scipy.optimize import minimize, approx_fprime, differential_evolution, Bounds
from datetime import datetime


class SviPowerSurface:
    def __init__(self, y: np.array, x: np.array, tau: np.array):
        self.y = y
        self.x = x
        self.tau = tau
        self.n, self.p = self.y.shape
        self.svi_k = 5
        self.curve_k = 3
        self._k_svi_params = self.svi_k * self.curve_k

        self.dgr_u2 = 1
        self.dgr_v2 = 1

        self._idx_a = np.arange(self.curve_k)
        self._idx_b = np.arange(self.curve_k, 2*self.curve_k)
        self._idx_rho = np.arange(2*self.curve_k, 3*self.curve_k)
        self._idx_m = np.arange(3*self.curve_k, 4*self.curve_k)
        self._idx_s2 = np.arange(4*self.curve_k, self._k_svi_params)
        self._idx_u2 = np.arange(self._k_svi_params, self._k_svi_params + self.dgr_u2 + 1)
        self._idx_v2 = np.arange(self._k_svi_params + self.dgr_u2 + 1, self._k_svi_params + self.dgr_u2 + self.dgr_v2 + 1)

        self.loc_notna = np.invert(np.isnan(self.vec(self.y))).flatten()
        self._nobs = self.loc_notna.sum()
        self._S = np.eye(self.n*self.p)[self.loc_notna]

        self._tau_pillars = np.unique(self.tau)
        self._x_pillars = np.unique(self.x)

    @staticmethod
    def vec(A):
        return A.reshape((-1, 1), order="F")

    @staticmethod
    def _polynomial(tau, betas, d):
        result = np.zeros_like(tau)
        for i in range(d+1):
            result += betas[i] * tau**i
        return result

    @staticmethod
    def surface(x, a, b, rho, m, s2):
        tmp = x - m
        return a + b * (rho * tmp + (tmp ** 2 + s2) ** .5)

    @staticmethod
    def surface2(x, v, b, rho, x_star, lmbda):
        tmp = x - x_star
        eta = np.sqrt(tmp**2 - 2*rho*tmp*lmbda + lmbda**2)
        return v + b*(rho*tmp + eta - lmbda)


    def _get_inits_mle(self):
        func = lambda params: self._objf_l2(params) / self._nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        # np.random.seed(5)
        inits = np.random.normal(scale=0., size=self._k_svi_params)
        optim = minimize(fun=func, x0=inits, method='L-BFGS-B', jac=grad, options={'maxiter':2000}, bounds=self._bounds)
        zeros = np.zeros(self._k_variance_params)
        zeros[0] += np.log(optim.fun)

        return np.random.normal(np.concatenate((optim.x, zeros), axis=0), scale=0.001)

        # return np.random.normal(np.concatenate((inits, zeros), axis=0), scale=0.1)

    def _set_bounds(self):
        self._bounds = []
        self._bounds += [(-5, -2), (-3, -0.5), (-1, 0)]
        self._bounds += [(1, 7), (-3, 0), (-7, -3)]
        self._bounds += [(-3, 1), (-2, 1), (-3, 0)]
        self._bounds += [(-1, 1), (-1, 1), (-2, 0)]
        self._bounds += [(-6, -1), (-3, 0), (-2, 0)]


    def _config(self):
        self._k_variance_params = self.dgr_v2 + self.dgr_u2 + 1 if self.mle else 0
        self._k_params = self._k_svi_params + self._k_variance_params
        self._set_bounds()
        self._objf = self._objf_mle if self.mle else self._objf_l2

        if self._inits is None:
            self._inits = self._get_inits_mle() if self.mle else np.random.normal(scale=0.1, size=self._k_params)

        if self.mle:
            self._bounds += [[-30, 5] for i in range(self._k_variance_params)]

    def fit(self, mle=False, inits=None):
        self._inits = inits
        self.mle = mle
        self._config()
        scaling = -1 if self.mle else 1
        func = lambda params: scaling * self._objf(params, False) / self._nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        if self.mle:
            self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad, bounds=self._bounds,
                                           options={'maxiter':2000})
        else:
            self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad, bounds=self._bounds,
                                           options={'maxiter':2000})
        self._rmspe = self._objf(self._optim_results.x, True)



    @staticmethod
    def _curve(tau, beta_min, beta_max, gamma, link_func):
       return link_func(beta_min) + (link_func(beta_max) - link_func(beta_min)) * tau ** np.exp(gamma)

    @staticmethod
    def _min1_plus1_to_R(a):
        return np.log((1 + a) / (1 - a)) / 2

    @staticmethod
    def _R_to_min1_plus1(a):
        return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))

    @staticmethod
    def _pos_to_R(a):
        return np.log(a)

    @staticmethod
    def _R_to_pos(a):
        return np.exp(a)

    @staticmethod
    def _R_to_R(a):
        return a


    def _params_to_matrices(self, params, tau, tau_pillars, x_pillars, mle:bool=False):
        a = self._curve(tau, *params[self._idx_a], self._R_to_pos)
        b = self._curve(tau, *params[self._idx_b], self._R_to_pos)
        rho = self._curve(tau, *params[self._idx_rho], self._R_to_min1_plus1)
        m = self._curve(tau, *params[self._idx_m], self._R_to_R)
        s2 = self._curve(tau, *params[self._idx_s2], self._R_to_pos)

        if not mle:
            return a, b, rho, m, s2
        else:
            U = np.diag(self._R_to_pos(self._polynomial(tau_pillars, params[self._idx_u2], self.dgr_u2)))
            tmp = np.concatenate((np.zeros(1), params[self._idx_v2]), axis=0)
            V = np.diag(self._R_to_pos(self._polynomial(x_pillars, tmp, self.dgr_v2)))
            return a, b, rho, m, s2, U, V

    def _objf_l2(self, transformed_params, return_rmspe:bool=False):
        args = self._params_to_matrices(transformed_params, self.tau, self._tau_pillars, self._x_pillars, False)
        y_hat = self.surface(self.x, *args)
        eps = self.vec(self.y - y_hat)[self.loc_notna]
        sse = np.sum(eps**2).flatten()[0]
        msre = ((self.vec(self.y/y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) *100
        return sse if not return_rmspe else rmspe

    def _objf_mle(self, transformed_params, return_rmspe: bool = False):
        a, b, rho, m, s2, U, V = self._params_to_matrices(transformed_params, self.tau, self._tau_pillars, self._x_pillars, True)
        y_hat = self.surface(self.x, a, b, rho, m, s2)
        eps = self.vec(self.y - y_hat)[self.loc_notna]
        Sigma = self._S @ np.kron(U, V) @ self._S.N
        diagSigma = np.diag(Sigma)
        invSigma = np.diag(1 / diagSigma)
        logdetSigma = np.log(diagSigma).sum()
        LL = - 0.5 * (self._nobs * np.log(2 * np.pi) + logdetSigma + eps.N @ invSigma @ eps)
        msre = ((self.vec(self.y / y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) * 100
        return LL.flatten()[0] if not return_rmspe else rmspe




