import config.logpoly
import numpy as np
from mpmath import mpf
import mpmath
import sys

def scale_data(data, min_value, max_value):
    return ((data - min_value) / (max_value - min_value)) * \
           ((1-(config.logpoly.right_margin + config.logpoly.left_margin)) * (config.logpoly.x_ubound - config.logpoly.x_lbound)) + \
            config.logpoly.x_lbound + (config.logpoly.left_margin * (config.logpoly.x_ubound - config.logpoly.x_lbound))


def mp_log_sum_exp(a):
    mx = np.max(a)
    res = mx + mpmath.log( np.sum( [mpmath.exp(a_i) for a_i in (a - mx)] ) )
    return res


def mp_compute_poly(x, theta):

    if not isinstance(x, np.ndarray):
        if type(x) == list:
            x = np.array(x)
        else:
            x = np.array([x])

    k = theta.size - 1
    n = x.size

    theta2d = np.tile(theta.reshape([1,-1]), [n,1])
    x_mpf = np.array([mpf(x_i) for x_i in x])

    tmp = np.tile(np.array([[mpf(1)]]), [n, k+1])
    for i in range(1,k+1):
        tmp[:, i] = tmp[:, i-1] * x_mpf

    # x2d = np.tile(np.array([mpf(x_i) for x_i in x]).reshape([n,1]), [1, k+1])
    # exp2d = np.tile(np.arange(k + 1).reshape([1, k + 1]), [n,1])
    #tmp = (x2d ** exp2d)

    return np.sum(tmp * theta2d, axis=1)


def mp_compute_SS(x, k):

    if config.logpoly.verbose:
        print('mp_compute_SS start')
        sys.stdout.flush()

    n = x.size

    x_mpf = np.array([mpf(x_i) for x_i in x])

    tmp = np.tile(np.array([[mpf(1)]]), [n, k + 1])
    for i in range(1, k + 1):
        tmp[:, i] = tmp[:, i - 1] * x_mpf

    result = np.sum(tmp, axis=0)
    if config.logpoly.verbose:
        print('mp_compute_SS finish')
        sys.stdout.flush()
    return result

def mp_log_integral_exp( log_func, theta):

    x_lbound, x_ubound = config.logpoly.x_lbound, config.logpoly.x_ubound

    p1 = mpf(x_lbound)
    p2 = mpf(x_ubound)
    l = p2 - p1
    parts = config.logpoly.mp_log_integral_exp_parts
    delta = l/parts
    points = np.arange(p1,p2,delta)
    if len(points) < parts + 1:
        points = np.concatenate([points, [p2]])

    f = log_func(points, theta)
    f = np.concatenate([f, f[1:-1]])

    intg = mp_log_sum_exp(f) + mpmath.log(l/parts) - mpmath.log(2)
    return intg


def mp_moments(func, max_power, x_lbound, x_ubound):

    buff = np.tile( np.array([mpf(0)]).reshape([1]), [max_power])

    p1 = mpf(x_lbound)
    p2 = mpf(x_ubound)
    l = p2 - p1
    parts = config.logpoly.mp_integral_parts
    delta = l/parts
    points = np.arange(p1, p2, delta)
    if len(points) < parts + 1:
        points = np.concatenate([points, [p2]])

    f = func(points)

    for j in range(max_power):
        tmp = 2 * np.sum(f[1:-1]) + f[0] + f[-1]
        buff[j] = tmp * (l/parts) / 2
        f = points * f
    return buff
