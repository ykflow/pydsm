import numpy as np
from scipy.optimize import minimize, approx_fprime, differential_evolution, Bounds
from datetime import datetime


class SeasonalNSPolySurface:
    def __init__(self, t, tau: np.array, x: np.array, y: np.array):
        self.t = t
        self.y = y
        self.x = x
        self.tau = tau
        self.n, self.p = self.y.shape
        self._t = self.day_of_year(self.t)

        self.ns_k = 3
        self.curve_k = 3
        self._k_ns_params = self.ns_k * self.curve_k + 1 + 4

        self.dgr_u2 = 2
        self.dgr_v2 = 2

        self._idx_a1g1 = np.arange(self.curve_k)
        self._idx_a2g2 = np.arange(self.curve_k, 2*self.curve_k)
        self._idx_a3g3 = np.arange(2*self.curve_k, 3*self.curve_k)
        self._idx_lmbda = np.arange(3*self.curve_k, 3 * self.curve_k + 1)
        self._idx_alphas = np.arange(3*self.curve_k+1, self._k_ns_params)

        self._idx_u2 = np.arange(self._k_ns_params, self._k_ns_params + self.dgr_u2 + 1)
        self._idx_v2 = np.arange(self._k_ns_params + self.dgr_u2 + 1, self._k_ns_params + self.dgr_u2 + self.dgr_v2 + 1)

        self.loc_notna = np.invert(np.isnan(self.vec(self.y))).flatten()
        self._nobs = self.loc_notna.sum()
        self._S = np.eye(self.n*self.p)[self.loc_notna]

        self._vec_x = self.vec(self.x)
        self._vec_tau = self.vec(self.tau)
        self._vec_y = self.vec(self.y)

        self._tau_pillars = np.unique(self.tau)
        self._x_pillars = np.unique(self.x)


    @staticmethod
    def day_of_year(t: datetime):
        return ((t - datetime(t.year, 1, 1)).days + 1) / 365

    @staticmethod
    def vec(A):
        return A.reshape((-1, 1), order="F")

    @staticmethod
    def _polynomial(tau, betas, d):
        result = np.zeros_like(tau)
        for i in range(d+1):
            result += betas[i] * tau**i
        return result


    def surface(self, tau, x, a1g1, a2g2, a3g3, lmbda, alpha):
        tmp1 = lmbda*tau
        tmp2 = (1 - np.exp(-tmp1))/tmp1
        tmp3 = tmp1 - np.exp(-tmp1)

        T = self.vec(self._t + tau)
        p1 = 2 * np.pi * T
        p2 = 4 * np.pi * T
        theta = alpha[0] * np.sin(p1) + alpha[1] * np.cos(p1) + alpha[2] * np.sin(p2) + alpha[3] * np.cos(p2)

        return a1g1 + a2g2*tmp2 + a3g3*tmp3 + theta

    def surface_deseasoned(self, tau, x, a, b, rho, m, s2, alpha):
        tmp = x - m
        T = self._t + tau
        p1 = 2 * np.pi * T
        p2 = 4 * np.pi * T
        theta = alpha[0] * np.sin(p1) + alpha[1] * np.cos(p1) + alpha[2] * np.sin(p2) + alpha[3] * np.cos(p2)
        return a + b * (rho * tmp + (tmp ** 2 + s2) ** .5)

    def seasons(self, tau, x, a, b, rho, m, s2, alpha):
        T = self._t + tau
        p1 = 2 * np.pi * T
        p2 = 4 * np.pi * T
        theta = alpha[0] * np.sin(p1) + alpha[1] * np.cos(p1) + alpha[2] * np.sin(p2) + alpha[3] * np.cos(p2)
        return theta*b



    def _get_inits_mle(self):
        func = lambda params: self._objf_l2(params) / self._nobs
        grad = lambda params: approx_fprime(params, func, 6.5e-6)
        # np.random.seed(5)
        inits = np.random.normal(scale=.1, size=self._k_ns_params)
        optim = minimize(fun=func, x0=inits, method='L-BFGS-B', jac=grad, options={'maxiter':2000}, #bounds=self._bounds
                         )
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
        self._bounds += [(-4, 4), (-4, 4), (-4, 4), (-4, 4)]


    def _config(self):
        self._k_variance_params = self.dgr_v2 + self.dgr_u2 + 1 if self.mle else 0
        self._k_params = self._k_ns_params + self._k_variance_params
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
            self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad, #bounds=self._bounds,
                                           options={'maxiter':2000})
        else:
            self._optim_results = minimize(fun=func, x0=self._inits, method='L-BFGS-B', jac=grad, #bounds=self._bounds,
                                           options={'maxiter':2000})
        self._rmspe = self._objf(self._optim_results.x, True)



    @staticmethod
    def _curve(x, params):
       return params[0] * np.exp(params[1]*x + params[2]*x**2 )

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


    def _params_to_matrices(self, params,  x, tau_pillars, x_pillars, mle:bool=False):
        a1g1 = self._curve(x, params[self._idx_a1g1])
        a2g2 = self._curve(x, params[self._idx_a2g2])
        a3g3 = self._curve(x, params[self._idx_a3g3])
        lmbda = np.array([1]) #self._R_to_pos(params[self._idx_lmbda])
        alphas = params[self._idx_alphas]

        if not mle:
            return a1g1, a2g2, a3g3, lmbda, alphas
        else:
            U = np.diag(self._R_to_pos(self._polynomial(tau_pillars, params[self._idx_u2], self.dgr_u2)))
            tmp = np.concatenate((np.zeros(1), params[self._idx_v2]), axis=0)
            V = np.diag(self._R_to_pos(self._polynomial(x_pillars, tmp, self.dgr_v2)))
            return a1g1, a2g2, a3g3, lmbda, alphas, U, V

    def _objf_l2(self, transformed_params, return_rmspe:bool=False):
        args = self._params_to_matrices(transformed_params, self._vec_x, self._tau_pillars, self._x_pillars, False)
        y_hat = self.surface(self._vec_tau, self._vec_x, *args)

        eps = (self._vec_y - y_hat)[self.loc_notna]
        sse = np.sum(eps**2).flatten()[0]
        msre = (((self._vec_y/y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) *100
        return sse if not return_rmspe else rmspe

    def _objf_mle(self, transformed_params, return_rmspe: bool = False):
        a1g1, a2g2, a3g3, lmbda, alphas, U, V = self._params_to_matrices(transformed_params, self._vec_x, self._tau_pillars, self._x_pillars, True)
        y_hat = self.surface(self._vec_tau, self._vec_x, a1g1, a2g2, a3g3, lmbda, alphas)

        eps = (self._vec_y - y_hat)[self.loc_notna]
        Sigma = self._S @ np.kron(U, V) @ self._S.N
        diagSigma = np.diag(Sigma)
        invSigma = np.diag(1 / diagSigma)
        logdetSigma = np.log(diagSigma).sum()
        LL = - 0.5 * (self._nobs * np.log(2 * np.pi) + logdetSigma + eps.N @ invSigma @ eps)

        msre = (((self._vec_y / y_hat)[self.loc_notna] - 1) ** 2).mean()
        rmspe = np.sqrt(msre) * 100
        return LL.flatten()[0] if not return_rmspe else rmspe
#