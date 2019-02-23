import config.logpoly
import numpy as np
import tensorflow as tf

# import tensorflow as tf
# from scipy import integrate

def scale_data(data, min_value, max_value):
    return ((data - min_value) / (max_value - min_value)) * (
            0.9 * (config.logpoly.x_ubound - config.logpoly.x_lbound)) + config.logpoly.x_lbound + (
                   0.05 * (config.logpoly.x_ubound - config.logpoly.x_lbound))


def compute_poly(x, theta):

    k = theta.shape[0]
    return tf.reduce_sum(
        tf.multiply(tf.pow(tf.tile(tf.reshape(x, [-1, 1]), [1, k]),
                           tf.constant(np.arange(k), dtype=tf.float32)),
                    theta))


def compute_SS(x, k, theta=None):
    n = x.size

    if theta is None:
        p = np.ones([n, 1])
    elif theta.shape[0] == 0:
        p = np.ones([n, 1])
    else:
        p = compute_poly(x, theta)

    exponent = np.tile(np.arange(k + 1).reshape([1, -1]), [n, 1])
    base = np.tile(x.reshape([-1, 1]), [1, k + 1])
    coef = np.tile(p.reshape([-1, 1]), [1, k + 1])

    return np.sum((base ** exponent) * coef, axis=0)

def log_integral_exp(log_func, theta):
    parts = 20000
    delta = (config.logpoly.x_ubound - config.logpoly.x_lbound)/parts
    points = tf.constant(np.arange(config.logpoly.x_lbound, config.logpoly.x_ubound + delta/2, delta), dtype=tf.float32)
    f = log_func(points, theta)
    return tf.subtract( tf.add( tf.reduce_logsumexp(f), tf.log(delta)), tf.log(2))


def compute_log_likelihood(SS, theta, n):
    if len(theta.shape) == 1:
        theta = theta.reshape([1, -1])
    # if len(theta.shape) > 1:
    #    theta = theta[-1,:]

    # logZ = log_integral_exp( compute_poly, theta, previous_critical_points)
    logZ = log_integral_exp(compute_poly, theta)
    ll = -n * logZ + np.inner(theta[-1, :].reshape([-1, ]), SS.reshape([-1, ]))

    return ll
