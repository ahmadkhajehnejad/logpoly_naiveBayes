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

    if len(theta.shape) == 1:
        theta = theta.reshape([1, -1])

    k = theta.shape[1] - 1
    t = theta.shape[0]
    n = x.size

    theta3d = np.tile(np.expand_dims(theta, 2), [1, 1, n])
    x3d = np.tile(np.array([mpf(x_i) for x_i in x]).reshape([1, 1, n]), [t, k + 1, 1])
    exp3d = np.tile(np.arange(k + 1).reshape([1, k + 1, 1]), [t, 1, n])

    return np.prod(np.sum((x3d ** exp3d) * theta3d, axis=1), axis=0)

    # result = [None] * n
    # for n_i in range(n):
    #     sums = [None] * t
    #     for t_i in range(t):
    #         tmp = [None] * (k + 1)
    #         for k_i in range(k + 1):
    #             tmp[k_i] = mpmath.power(x[n_i], k_i) * theta[t_i, k_i]
    #         sums[t_i] = mpmath.fsum(tmp)
    #     result[n_i] = mpmath.fprod(sums)
    #
    # return result


def mp_compute_SS(x, k, theta=None):
    print('mp_compute_SS start')
    sys.stdout.flush()
    n = x.size

    if theta is None:
        p = np.ones([n])
    elif theta.shape[0] == 0:
        p = np.ones([n])
    else:
        p = mp_compute_poly(x, theta)

    exponent = np.tile(np.arange(k + 1).reshape([1, -1]), [n, 1])
    base = np.tile(np.array([mpf(x_i) for x_i in x]).reshape([-1, 1]), [1, k + 1])
    coef = np.tile(p.reshape([-1, 1]), [1, k + 1])

    result = np.sum((base ** exponent) * coef, axis=0)
    print('mp_compute_SS finish')
    sys.stdout.flush()
    return result

    # result = [None] * (k+1)
    # for k_i in range(k+1):
    #     tmp = [None] * n
    #     for n_i in range(n):
    #         tmp[n_i] = p[n_i] * mpmath.power(x[n_i], k_i)
    #     sys.stdout.flush()
    #     result[k_i] = mpmath.fsum(tmp)
    #
    # print('mp_compute_SS finish')
    # sys.stdout.flush()
    # return result

def mp_log_integral_exp( log_func, theta):

    x_lbound, x_ubound = config.logpoly.x_lbound, config.logpoly.x_ubound
    if len(theta.shape) == 1:
        theta = theta.reshape([1,-1])
    all_theta = theta[0, :].reshape([-1]).copy()
    d = all_theta.size - 1

    for i in range(1,len(theta)):
        tmp = mpmath.matrix(theta[i,:].reshape([-1,1])) * mpmath.matrix(all_theta.reshape([1,-1]))
        all_theta = np.array([mpf(0) for _ in range(all_theta.size + d)])
        for i1 in range(tmp.rows):
            for i2 in range(tmp.cols):
                all_theta[i1+i2] += tmp[i1, i2]

    d_all = all_theta.size - 1
    derivative_poly_coeffs = np.flip(all_theta[1:] * np.arange(1,d_all+1), axis=0)


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
        parts = 100
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
        parts = 100
        delta = l/parts
        points = np.arange(p1, p2, delta)
        if len(points) < parts + 1:
            points = np.concatenate([points, [p2]])

        f = func(points)
        f = np.concatenate([f, f[1:-1]])

        buff[i] = np.sum(f) * (l/parts) / 2
    return np.sum(buff)
