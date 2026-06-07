import numpy as np
from scipy.optimize import minimize, approx_fprime, differential_evolution, Bounds
from datetime import datetime


class SviPolySurface:
    def __init__(self, y: np.array, x: np.array, tau: np.array, degrees: dict):
        self.y = y
        self.x = x
        self.tau = tau
        self.n, self.p = self.y.shape
        self._k = 5
        self.dgr_a = degrees['a']
        self.dgr_b = degrees['b']
        self.dgr_rho = degrees['rho']
        self.dgr_m = degrees['m']
        self.dgr_s2 = degrees['s2']
        self.dgr_u2 = degrees['u2']
        self.dgr_v2 = degrees['v2']

        self.loc_notna = np.invert(np.isnan(self.vec(self.y))).flatten()
        self._nobs = self.loc_notna.sum()
        self._S = np.eye(self.n*self.p)[self.loc_notna]

        self._k_svi_params = self.dgr_a + self.dgr_b + self.dgr_rho + self.dgr_m + self.dgr_s2 + self._k
        self._idx_a = np.arange(self.dgr_a+1)
        self._idx_b = np.arange(self.dgr_a+1, self.dgr_a+self.dgr_b+2)
        self._idx_rho = np.arange(self.dgr_a+self.dgr_b+2, self.dgr_a+self.dgr_b+self.dgr_rho+3)
        self._idx_m = np.arange(self.dgr_a+self.dgr_b+self.dgr_rho+3, self.dgr_a+self.dgr_b+self.dgr_rho+self.dgr_m+4)
        self._idx_s2 = np.arange(self.dgr_a + self.dgr_b + self.dgr_rho + self.dgr_m + 4, self._k_svi_params)
        self._idx_u2 = np.arange(self._k_svi_params, self._k_svi_params+self.dgr_u2+1)
        self._idx_v2 = np.arange(self._k_svi_params+self.dgr_u2+1, self._k_svi_params+self.dgr_u2+self.dgr_v2+1)

        self._tau_pillars = np.unique(self.tau)
        self._x_pillars = np.unique(self.x)

    @staticmethod
    def vec(A):
        return A.reshape((-1, 1), order="F")

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
        inits = np.random.normal(scale=0.1, size=self._k_svi_params)
        optim = minimize(fun=func, x0=inits, method='L-BFGS-B', jac=grad, options={'maxiter':500})
        zeros = np.zeros(self._k_variance_params)
        if len(zeros) ==1:
            zeros -= 5
        else:
            zeros[:-1] -= 5
        return np.random.normal(np.concatenate((optim.x, zeros), axis=0), scale=0.1)

    def _set_bounds(self):
        bounds_a = (-4, 2)
        bounds_b = (-2, 2)
        bounds_rho = (-3, 3)
        bounds_m = (-1, 1)
        bounds_s2 = (-6, 2)

        self._bounds = []
        self._bounds += [[bounds_a[0], bounds_a[1]] for i in range(self.dgr_a+1)]
        self._bounds += [[bounds_b[0], bounds_b[1]] for i in range(self.dgr_b + 1)]
        self._bounds += [[bounds_rho[0], bounds_rho[1]] for i in range(self.dgr_rho + 1)]
        self._bounds += [[bounds_m[0], bounds_m[1]] for i in range(self.dgr_m + 1)]
        self._bounds += [[bounds_s2[0], bounds_s2[1]] for i in range(self.dgr_s2 + 1)]

    def _config(self):
        self._k_variance_params = self.dgr_v2 + self.dgr_u2 + 1 if self.mle else 0
        self._k_params = self._k_svi_params + self._k_variance_params
        self._set_bounds()
        self._objf = self._objf_mle if self.mle else self._objf_l2

        if self.inits is None:
            self._inits = self._get_inits_mle() if self.mle else np.random.normal(scale=0.1, size=self._k_params)

        if self.mle:
            self._bounds += [[-10, 10] for i in range(self._k_variance_params)]


    def fit(self, mle=False, inits=None):
        self.inits = inits
        self.mle = mle
        self._config()
        scaling = -1 if self.mle else 1
        func = lambda params: scaling * self._objf(params, False) / self._nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        if self.mle:
            self._optim_results = minimize(fun=func, x0=self._inits, method='SLSQP', jac=grad, #bounds=self._bounds,
                                           options={'maxiter':500})
        else:
            self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad, #bounds=self._bounds,
                                           options={'maxiter':500})
        self._rmspe = self._objf(self._optim_results.x, True)

    @staticmethod
    def _polynomial(tau, betas, d):
        result = np.zeros_like(tau)
        for i in range(d+1):
            result += betas[i] * tau**i
        return result

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

    def _params_to_matrices(self, params, tau, tau_pillars, x_pillars, mle:bool=False):
        a = self._R_to_pos(self._polynomial(tau, params[self._idx_a], self.dgr_a))
        b = self._R_to_pos(self._polynomial(tau, params[self._idx_b], self.dgr_b))
        rho = self._R_to_min1_plus1(self._polynomial(tau, params[self._idx_rho], self.dgr_rho))
        m = self._polynomial(tau, params[self._idx_m], self.dgr_m)
        s2 = self._R_to_pos(self._polynomial(tau, params[self._idx_s2], self.dgr_s2))

        if not mle:
            return a, b, rho, m, s2
        else:
            U = np.diag(self._R_to_pos(self._polynomial(tau_pillars, params[self._idx_u2], self.dgr_u2)))
            tmp = np.concatenate((np.zeros(1), params[self._idx_v2]), axis=0)
            V = np.diag(self._R_to_pos(self._polynomial(x_pillars, tmp, self.dgr_v2)))
        return a, b, rho, m, s2, U, V

    def _objf_l2(self, transformed_params, return_mse:bool=False):
        args = self._params_to_matrices(transformed_params, self.tau, self._tau_pillars, self._x_pillars, False)
        y_hat = self.surface(self.x, *args)
        eps = self.vec(self.y - y_hat)[self.loc_notna]
        sse = np.sum(eps**2).flatten()[0]
        msre = ((self.vec(self.y/y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) *100
        return sse if not return_mse else rmspe

    def _objf_mle(self, transformed_params, return_rmspe:bool=False):
        a, b, rho, m, s2, U, V = self._params_to_matrices(transformed_params, self.tau, self._tau_pillars, self._x_pillars, True)
        y_hat = self.surface(self.x, a, b, rho, m, s2)
        eps = self.vec(self.y - y_hat)[self.loc_notna]
        Sigma = self._S @ np.kron(U, V) @ self._S.N
        diagSigma = np.diag(Sigma)
        invSigma = np.diag(1/diagSigma)
        logdetSigma = np.log(diagSigma).sum()
        LL = - 0.5*(self._nobs * np.log(2*np.pi) + logdetSigma + eps.N @ invSigma @ eps)
        msre = ((self.vec(self.y/y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) * 100
        return LL.flatten()[0] if not return_rmspe else rmspe