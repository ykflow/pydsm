import warnings
warnings.filterwarnings('ignore')
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
import pandas as pd
import timeit
import torch


p = 10
k = 4
x = np.array([-0.4, -0.4, -0.3, -0.3, 0, 0, .3, .3, 0.4, 0.4, ]).reshape(p, 1)
tau = np.array([1, 12, 1, 12, 1, 12, 1, 12, 1, 12]).reshape(p, 1) / 12

eta = -0.225
# eta = 0
eta_tau = np.exp(eta*tau)

def carr_wu_iv(kappa, omega, nu, rho, gamma, x, tau):
    tmp = 1 - 2 * kappa * tau - gamma * tau
    a = -2 * tmp / (omega ** 2 * tau ** 2)
    b = - rho * nu / omega
    c = ((1 - rho ** 2) * (nu ** 2) / (omega ** 2)) + tmp ** 2 / (omega ** 4 * tau ** 2)
    I2 = a + (2 / tau) * ((x - b) ** 2 + c)**.5
    return I2

def map_moments(omega_star, nu_star, rho_star):
    omega = np.exp(omega_star)
    nu = np.exp(nu_star)
    rho = (np.exp(rho_star) - np.exp(-rho_star)) / (np.exp(rho_star) + np.exp(-rho_star))
    gamma = rho * omega * nu
    return omega, nu, rho, gamma

def carr_wu_standard(f):
    kappa, omega_star, nu_star, rho_star = f.flatten()
    omega, nu, rho, gamma = map_moments(omega_star, nu_star, rho_star)
    return carr_wu_iv(kappa*eta_tau, omega*eta_tau, nu, rho, gamma, x, tau)


def jac_carr_wu_standard(f):
    kappa, omega_star, nu_star, rho_star = f.flatten()
    omega, nu, rho, gamma = map_moments(omega_star, nu_star, rho_star)
    kappa = kappa * eta_tau
    omega = omega * eta_tau

    tmp = 1 - 2 * kappa * tau - gamma * tau
    a = -2 * tmp / (omega ** 2 * tau ** 2)
    b = - rho * nu / omega
    c = ((1 - rho ** 2) * (nu ** 2) / (omega ** 2)) + tmp ** 2 / (omega ** 4 * tau ** 2)
    denom = tau * ((x - b) ** 2 + c)**.5
    z1 = 4/(omega**2 * tau) - 4*tmp/(omega**4 * tau**2 * ((x - b) ** 2 + c)**.5)
    z2 = -2*(1/omega * (a + b/tau)
          +(
            - 1*(b*((x-b)/omega + tmp/(omega**3 * tau)))
            + 1*((1-rho**2) * (nu**2 / omega**3) + 2 * tmp**2 / (omega**5 * tau**2))
          )/denom) * omega

    z3 = 2*(rho/(omega * tau) + (rho*(x-b)/omega + nu*(1-rho**2)/omega**2 - rho* tmp/(omega**3 * tau)
                                   )/denom) * nu

    z4 = 2 * (nu/(omega*tau) + nu*(x/omega - tmp/(omega**3 * tau))/denom) * (4 * np.exp(2*rho_star)/(np.exp(2*rho_star)+1)**2)


    return np.concatenate((z1*eta_tau, z2*eta_tau, z3, z4), axis=1)




J = jacobian(carr_wu_standard)

params = np.array([-.95, -.45, -.35, .333])
Jauto = np.array(J(params)[:, 0,:])
Jtrue = jac_carr_wu_standard(params)

print(np.allclose(Jtrue, Jauto))

Japprox = np.zeros((p, k))
eps = 5e-6
for i in range(k):
    h = np.zeros(params.shape)
    h[i] += eps
    hp = params + h
    hm = params - h
    Ji = (carr_wu_standard(hp) - carr_wu_standard(hm))/(2*eps)
    Japprox[:, i] = Ji.flatten()

print(Japprox)


# x = -0.35
# tau =0.8
# gamma = .5
#
# def h(tau, h_min, h_max, gamma, link):
#     return link(h_min) + (link(h_max) -link(h_min)) * tau**gamma
#
# def i(a):
#     return a
#
# def ts_svi(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     a = h(tau, a_min, a_max, gamma, torch.exp)
#     b = h(tau, b_min, b_max, gamma, torch.exp)
#     rho = h(tau, rho_min, rho_max, gamma, torch.tanh)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, torch.exp)
#
#     tmp0 = (x - kappa)
#     return a + b*(rho * tmp0 + (tmp0**2 + s2)**.5)
#
# def dZda_min(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     return (1 - tau**gamma)*np.exp(a_min)
#
# def dZda_max(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     return (tau**gamma)*np.exp(a_max)
#
# def dZdb_min(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     rho = h(tau, rho_min, rho_max, gamma, np.tanh)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, np.exp)
#     tmp0 = (x - kappa)
#     tmp1 = (rho * tmp0 + (tmp0**2 + s2)**.5)
#     return tmp1*(1 - tau**gamma)*np.exp(b_min)
#
# def dZdb_max(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     rho = h(tau, rho_min, rho_max, gamma, np.tanh)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, np.exp)
#     tmp0 = (x - kappa)
#     tmp1 = (rho * tmp0 + (tmp0**2 + s2)**.5)
#     return tmp1*(tau**gamma)*np.exp(b_max)
#
# def dZdrho_min(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     b = h(tau, b_min, b_max, gamma, np.exp)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     tmp0 = (x - kappa)
#     tmp1 = b*tmp0
#     tmp3 = (4*np.exp(2*rho_min))/((1 + np.exp(2*rho_min))**2)
#     return tmp1*(1- tau**gamma)*tmp3
#
# def dZdrho_max(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     b = h(tau, b_min, b_max, gamma, np.exp)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     tmp0 = (x - kappa)
#     tmp1 = b*tmp0
#     tmp3 = (4*np.exp(2*rho_max))/((1 + np.exp(2*rho_max))**2)
#     return tmp1*(tau**gamma)*tmp3
#
#
# def dZdkappa_min(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     b = h(tau, b_min, b_max, gamma, np.exp)
#     rho = h(tau, rho_min, rho_max, gamma, np.tanh)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, np.exp)
#
#     tmp0 = (x - kappa)
#     tmp1 = -b*(rho + tmp0/(tmp0**2 + s2)**.5)
#     return tmp1*(1 - tau**gamma)
#
# def dZdkappa_max(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     b = h(tau, b_min, b_max, gamma, np.exp)
#     rho = h(tau, rho_min, rho_max, gamma, np.tanh)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, np.exp)
#
#     tmp0 = (x - kappa)
#     tmp1 = -b*(rho+tmp0/(tmp0**2 + s2)**.5)
#     return tmp1*(tau**gamma)
#
#
# def dZds2_min(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     b = h(tau, b_min, b_max, gamma, np.exp)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, np.exp)
#
#     tmp0 = (x - kappa)
#     tmp1 = np.exp(s2_min)
#     tmp2 = 2*(tmp0**2 + s2)**.5
#     tmp3 = b*(tmp1/tmp2)
#     return tmp3*(1-tau**gamma)
#
# def dZds2_max(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     b = h(tau, b_min, b_max, gamma, np.exp)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, np.exp)
#
#     tmp0 = (x - kappa)
#     tmp1 = np.exp(s2_max)
#     tmp2 = 2*(tmp0**2 + s2)**.5
#     tmp3 = b*(tmp1/tmp2)
#     return tmp3*(tau**gamma)
#
#
#
# params = np.array([1., 2., 1., 2., 1., 2., 1., 2., 1., 2.])
# dZdf = jacobian(ts_svi)
#
# print(dZdf(params),)
#
# result = np.array([dZda_min(params), dZda_max(params), dZdb_min(params), dZdb_max(params),
#                    dZdrho_min(params), dZdrho_max(params), dZdkappa_min(params), dZdkappa_max(params),
#                    dZds2_min(params), dZds2_max(params)])
# print(result)
#
#
#
# from torch.autograd.functional import jacobian  as jt
# def ts_svi(params):
#     a_min, a_max, b_min, b_max, rho_min, rho_max, kappa_min, kappa_max, s2_min, s2_max = params
#     a = h(tau, a_min, a_max, gamma, torch.exp)
#     b = h(tau, b_min, b_max, gamma, torch.exp)
#     rho = h(tau, rho_min, rho_max, gamma, torch.tanh)
#     kappa = h(tau, kappa_min, kappa_max, gamma, i)
#     s2 = h(tau, s2_min, s2_max, gamma, torch.exp)
#
#     tmp0 = (x - kappa)
#     return a + b*(rho * tmp0 + (tmp0**2 + s2)**.5)
#
# # dZdf = jt(ts_svi, None)
# dZdf = jt(ts_svi, torch.from_numpy(params), create_graph=True)
# print(dZdf( torch.from_numpy(params)),)