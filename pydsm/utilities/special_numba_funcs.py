import numpy as np
from numpy.linalg import inv, pinv
from numba import jit


@jit(nopython=True, cache=True, error_model='numpy')
def gammaln(z):
    """Natural logarithm of the Gamma of scalar z"""

    coefs = np.array([
        57.1562356658629235, -59.5979603554754912,
        14.1360979747417471, -0.491913816097620199,
        .339946499848118887e-4, .465236289270485756e-4,
        -.983744753048795646e-4, .158088703224912494e-3,
        -.210264441724104883e-3, .217439618115212643e-3,
        -.164318106536763890e-3, .844182239838527433e-4,
        -.261908384015814087e-4, .368991826595316234e-5])

    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y = y + 1.
        ser = ser + coefs[j] / y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out

@jit(nopython=True, cache=True)
def jinv(A):
    x = np.copy(A)
    return inv(x)

@jit(nopython=True, cache=True)
def jpinv(A):
    return pinv(np.copy(A))

@jit(nopython=True, fastmath=True, cache=True)
def locate_missings(a):
    a_ = np.copy(a)
    nans = np.zeros((a_.shape))
    for i in range(len(a_)):
        
        if a_[i] == -9223372036854775808:
            nans[i] = True
        else:
            nans[i] = False
            
    loc_nans = np.argwhere(nans==1)[:, 0]
    n_nans = loc_nans.shape[0]
    loc_nans = np.copy(loc_nans).reshape(n_nans,1)
    
    loc_not_nans = np.argwhere(nans==0)[:, 0]
    n_not_nans = loc_not_nans.shape[0]
    loc_not_nans = np.copy(loc_not_nans).reshape(n_not_nans,1)
    return loc_nans, n_nans, loc_not_nans, n_not_nans

@jit(nopython=True, fastmath=True, cache=True)
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return np.copy(result)

@jit(nopython=True, fastmath=True,cache=True)
def mean(array, axis):
  return np_apply_along_axis(np.nanmean, axis, array)

@jit(nopython=True, fastmath=True, cache=True)
def var(array, axis):
  return np_apply_along_axis(np.nanvar, axis, array)

@jit(nopython=True, cache=True)
def first_n_mean(y, n_max=50):
    n, p, _ = y.shape
    means = np.zeros((p,1), dtype=np.float64) * np.nan
    for i in range(p):
        y_i = y[:, i].flatten()
        y_i = y_i[np.abs(y_i) >=0.] #skip missing
        means[i] = np.mean(y_i[:n_max])
    return means

@jit(nopython=True, cache=True)
def first_n_logVar(y, n_max=50):
    means = first_n_mean(y, n_max=n_max)
    n, p, _ = y.shape
    log_var = np.zeros((p,1), dtype=np.float64) * np.nan
    for i in range(p):
        y_i = y[:, i].flatten()
        y_i = y_i[np.abs(y_i) >=0]
        log_var[i] = np.log(np.var(y_i[:n_max] - means[i]))
    return log_var

@jit(nopython=True, cache=True)
def logdet(a):
    return np.linalg.slogdet(np.copy(a))[1]

@jit(nopython=True, cache=True)
def remove_effect_coding(F):
    F_tmp = np.copy(F)
    return np.where(F_tmp == -1, 0, F_tmp)

@jit(nopython=True, cache=True)
def locate_nonzero_scores(F):
    F_tmp = remove_effect_coding(np.copy(F))
    return F_tmp.sum(axis=0) != 0

@jit(nopython=True, cache=True)
def make_M_matrix(F):
    indicator = 1. * locate_nonzero_scores(np.copy(F))
    return np.diag(indicator) * 1.0  ##ensure float64 type


@jit(nopython=True, cache=True)
def random_normal(n):
    """Generate n random normal deviates."""

    u1 = np.random.random(n)
    u2 = np.random.random(n)
    r_squared = -2 * np.log(u1)
    r = np.sqrt(r_squared)
    theta = 2 * np.pi * u2
    x = r * np.cos(theta)
    z = np.empty(n, dtype=np.float64)
    z = x

    return z[:n]

# F = np.array([[1,1,0,0,
#                1,0,1,0,
#                1,0,0,1,
#                1, -1, -1, -1]], dtype=np.float64).reshape(4,4)
#
# F_dag = F[(0,3), :]
# print(F)
# print(F_dag)
# print(locate_nonzero_scores(F))
# print(locate_nonzero_scores(F_dag))
#
# print(make_M_matrix(F_dag))