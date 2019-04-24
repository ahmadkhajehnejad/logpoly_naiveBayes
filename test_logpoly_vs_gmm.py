import numpy as np
import matplotlib.pyplot as plt
from logpoly.model import Logpoly, _compute_log_likelihood, LogpolyModelSelector
from logpoly.tools import mp_compute_SS, scale_data
import config.logpoly
from Gaussian_mixture_model.model import GaussianMixtureModel
import sys
import pandas as pd



def generate_samples(n, pi, mu, sigma):

    k = pi.size
    z = np.random.choice(np.arange(k), size=n, replace=True, p=pi)

    samples = np.random.normal(0, 1, n) * sigma[z] + mu[z]

    return samples

if __name__ == '__main__':


    n = 100000
    ll = 0.2
    md = 0.7
    rr =  0.8
    samples = np.random.triangular(ll, md, rr, n)
    print(np.min(samples))
    print(np.max(samples))
    min_x = 0
    max_x = 1


    # n = 100000
    # ll = 0.2
    # rr =  0.8
    # samples = np.random.uniform(ll, rr, n)
    # print(np.min(samples))
    # print(np.max(samples))
    # min_x = 0
    # max_x = 1


    # n = 100000
    # samples = np.random.exponential(10, n)
    # print(np.min(samples))
    # print(np.max(samples))
    # min_x = np.min(samples)
    # max_x = np.max(samples)


    # pi_tmp = np.array([9, 2, 3, 4, 1, 7, 2])
    # pi = pi_tmp / np.sum(pi_tmp)
    # mu = np.array([1, 20, 50, 85, 130, 160, 80])
    # sigma = np.array([1, 2, 3, 1, 2, 3, 40])
    # n = 100000
    # min_x = -80
    # max_x = 250

    # pi = np.array([9/26, 2/26, 3/26, 4/26, 1/26, 7/26])
    # mu = np.array([1, 20, 50, 85, 130, 160])
    # sigma = np.array([1, 2, 3, 1, 2, 3])
    # n = 100000
    # min_x = -20
    # max_x = 180

    # pi = np.array([28/40, 3/40, 3/40, 3/40, 3/40])
    # mu = np.array([0, 50, 100, 150, 200])
    # sigma = np.array([1, 1, 1, 1, 1])
    # n = 100000
    # min_x = -50
    # max_x = 250

    # pi_tmp = np.array([1, 1, 1])
    # pi = pi_tmp / np.sum(pi_tmp)
    # mu = np.array([0, 10, 19])
    # sigma = np.array([10, 10, 10])
    # n = 100000
    # min_x = -50
    # max_x = 70

    # n = 100000
    # samples = 10 * (np.random.rand(n) - 0.5)**3
    # print(np.min(samples))
    # print(np.max(samples))
    # min_x = -1.5
    # max_x = 1.5



    # samples = generate_samples(n, pi, mu, sigma)

    print(np.min(samples), np.max(samples))

    scaled_samples = scale_data(samples, min_x, max_x)
    print(np.min(scaled_samples), np.max(scaled_samples))
    ticks = np.arange(min_x, max_x, 0.01)

    avgLL_true = np.mean(triangular_pdf(samples, ll, md, rr))

    # plt.plot([ll, ll, rr, rr], [0, 1 / (rr - ll), 1 / (rr - ll), 0], color='blue')
    plt.plot([ll, md, rr], [0, 2 / (rr - ll), 0], color='blue')
    # plt.hist(samples, bins=2000, density=True)
    # gmm_avgll = []
    for k, c in [(2,'red'), (3,'red'), (4,'red'), (5,'red'), (6,'red')]: #[(2,'orange'), (4,'red'), (6,'black')]:
        print('GMM - k:', k)
        gmm = GaussianMixtureModel(scaled_samples, num_components=k)
        print('gmm KL: ', avgLL_true - np.mean(gmm.logpdf(scaled_samples)))
        # gmm_avgll.append(np.sum(gmm.logpdf(scaled_samples)))
        sys.stdout.flush()

        y_ticks_gmm = np.exp(gmm.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))

        plt.plot(ticks, y_ticks_gmm, color=c)

    # print('GMM mean-avgll:', np.mean(gmm_avgll))

    for k, c in [(5,'green'), (8,'green'), (11,'green'), (14,'green'), (17,'green')]:
        logpoly = Logpoly()
        print('logpoly - k:', k)
        logpoly.fit(mp_compute_SS(scaled_samples, k), n)
        print('logpoly KL: ', avgLL_true - (logpoly.current_log_likelihood)/n)
        sys.stdout.flush()
        y_ticks_logpoly = np.exp(logpoly.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))
        plt.plot(ticks, y_ticks_logpoly, color=c)

    plt.show()
