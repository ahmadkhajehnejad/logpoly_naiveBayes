import config.logpoly
import numpy as np
from mpmath import mpf
import mpmath
import sys

def scale_data(data, min_value, max_value):
    return ((data - min_value) / (max_value - min_value)) * (
                0.9 * (config.logpoly.x_ubound - config.logpoly.x_lbound)) + config.logpoly.x_lbound + (
                         0.05 * (config.logpoly.x_ubound - config.logpoly.x_lbound))


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

    k = theta.size - 1
    derivative_poly_coeffs = np.flip(theta[1:] * np.arange(1,k+1))

    if config.logpoly.verbose:
        print('           poly_roots start')
        sys.stdout.flush()
    # r = np.roots(derivative_poly_coeffs)
    # r = r.real[np.abs(r.imag) < 1e-10]
    # r = np.unique(r[(r >= x_lbound) & (r <= x_ubound)])
    # r = np.array([mpf(r_i) for r_i in r])

    _tmp_ind = derivative_poly_coeffs != 0
    if not np.any(_tmp_ind):
        r = np.array([])
    else:
        derivative_poly_coeffs = derivative_poly_coeffs[np.argmax(_tmp_ind):]
        r = mpmath.polyroots(derivative_poly_coeffs, maxsteps=500, extraprec=mpmath.mp.dps)
        r = np.array([r_i for r_i in r if isinstance(r_i, mpf)])
        r = np.unique(r[ (r >= x_lbound) & (r <= x_ubound)])

    if config.logpoly.verbose:
        print('           poly_roots finish')
        sys.stdout.flush()

    if r.size > 0:
        br_points = np.unique(np.concatenate([np.array([mpf(x_lbound)]), r.reshape([-1]), np.array([mpf(x_ubound)])]))
    else:
        br_points = np.array([mpf(x_lbound), mpf(x_ubound)])

    buff = np.array([mpf(0) for _ in range(br_points.size -1)])
    for i in range(br_points.size - 1):
        p1 = br_points[i]
        p2 = br_points[i+1]
        l = p2 - p1
        parts = config.logpoly.mp_log_integral_exp_parts
        delta = l/parts
        points = np.arange(p1,p2,delta)
        if len(points) < parts + 1:
            points = np.concatenate([points, [p2]])

        f = log_func(points, theta)
        f = np.concatenate([f, f[1:-1]])

        buff[i] = mp_log_sum_exp(f) + mpmath.log(l/parts) - mpmath.log(2)
    return mp_log_sum_exp(buff), r


def mp_integral(func, roots, x_lbound, x_ubound):

    roots = np.unique(roots[(roots >= x_lbound) & (roots <= x_ubound)])

    if roots.size > 0:
        br_points = np.unique(np.concatenate([np.array([mpf(x_lbound)]), roots.reshape([-1]), np.array([mpf(x_ubound)])]))
    else:
        br_points = np.array([mpf(x_lbound), mpf(x_ubound)])

    buff = np.array([mpf(0) for _ in range(br_points.size -1)])
    for i in range(br_points.size - 1):
        # print('^', end='')
        # sys.stdout.flush()
        p1 = br_points[i]
        p2 = br_points[i+1]
        l = p2 - p1
        parts = config.logpoly.mp_integral_parts
        delta = l/parts
        points = np.arange(p1, p2, delta)
        if len(points) < parts + 1:
            points = np.concatenate([points, [p2]])

        f = func(points)
        f = np.concatenate([f, f[1:-1]])

        buff[i] = np.sum(f) * (l/parts) / 2
    return np.sum(buff)


def mp_moments(func, max_power, roots, x_lbound, x_ubound):

    roots = np.unique(roots[(roots >= x_lbound) & (roots <= x_ubound)])

    if roots.size > 0:
        br_points = np.unique(np.concatenate([np.array([mpf(x_lbound)]), roots.reshape([-1]), np.array([mpf(x_ubound)])]))
    else:
        br_points = np.array([mpf(x_lbound), mpf(x_ubound)])

    buff = np.tile( np.array([mpf(0)]).reshape([1,1]), [max_power, br_points.size-1])

    for i in range(br_points.size - 1):
        # print('^', end='')
        # sys.stdout.flush()
        p1 = br_points[i]
        p2 = br_points[i+1]
        l = p2 - p1
        parts = config.logpoly.mp_integral_parts
        delta = l/parts
        points = np.arange(p1, p2, delta)
        if len(points) < parts + 1:
            points = np.concatenate([points, [p2]])

        f = func(points)

        g = f
        for j in range(max_power):
            tmp = 2 * np.sum(g[1:-1]) + g[0] + g[-1]
            buff[j,i] = tmp * (l/parts) / 2
            g = points * g
    return np.sum(buff, axis=1)
