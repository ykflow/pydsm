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