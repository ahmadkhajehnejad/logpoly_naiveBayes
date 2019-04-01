import numpy as np
import matplotlib.pyplot as plt
from logpoly.model import Logpoly, _compute_log_likelihood
from logpoly.tools import mp_compute_SS, scale_data
import config.logpoly
import sys


def generate_samples(n, pi, mu, sigma):

    k = pi.size
    z = np.random.choice(np.arange(k), size=n, replace=True, p=pi)

    samples = np.random.normal(0, 1, n) * sigma[z] + mu[z]

    return samples


if __name__ == '__main__':

    pi = np.array([9/26, 2/26, 3/26, 4/26, 1/26, 7/26])
    mu = np.array([1, 20, 50, 85, 130, 160])
    sigma = np.array([1, 2, 3, 1, 2, 3])
    n = 100000
    min_x = -20
    max_x = 180

    samples = generate_samples(n, pi, mu, sigma)

    print(np.min(samples), np.max(samples))

    #n = 1000000
    #samples = samples[:n]
    scaled_samples = scale_data(samples, min_x, max_x)
    print(np.min(scaled_samples), np.max(scaled_samples))
    k = 18
    logpoly = Logpoly(k, factors_count=1, gradient_size=None)
    print('fit start')
    sys.stdout.flush()
    logpoly.fit(scaled_samples, n)
    print('fit finished')
    print('loglikelihood: ', logpoly.current_log_likelihood)
    sys.stdout.flush()
    ticks = np.arange(min_x, max_x, 0.1)
    y_ticks = np.exp(logpoly.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / 0.9)


    print(scale_data(ticks, min_x, max_x))

    print(np.sum(np.exp(logpoly.logpdf(np.arange(0, 1, 0.001)))) / 1000)
    print(ticks)

    plt.hist(samples, bins=2000, density=True)
    plt.plot(ticks, y_ticks, color='green')
    plt.show()