import numpy as np
from scipy.stats import norm


class BlackFuturesOptionsPricer:
    def __init__(self, F, K, tau, sigma2, r):
        self.F = F
        self.K = K
        self.tau = tau
        self.sigma2 = sigma2
        self.r = r

    def call_price(self):
        tmp = (self.sigma2 * self.tau) ** .5
        x = np.log(self.F/self.K)
        d1 = (x + (self.sigma2/2) * self.tau) / tmp
        d2 = d1 - tmp
        return np.exp(-self.r*self.tau) * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2))

    def put_price(self):
        tmp = (self.sigma2 * self.tau) ** .5
        x = np.log(self.F / self.K)
        d1 = (x + (self.sigma2 / 2) * self.tau) / tmp
        d2 = d1 - tmp
        return np.exp(-self.r * self.tau) * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))




# def black_call_price(F, x, r, tau, sigma2):
#     tmp = (sigma2 * tau) ** .5
#     d1 = (x + (sigma2/2) * tau) / tmp
#     d2 = d1 - tmp
#     return np.exp(-r*tau) * (F * norm.cdf(d1) - )
