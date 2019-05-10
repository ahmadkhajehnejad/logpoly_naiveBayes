import numpy as np
import matplotlib.pyplot as plt
from logpoly.model import Logpoly, _compute_log_likelihood, LogpolyModelSelector
from logpoly.tools import mp_compute_SS, scale_data
import config.logpoly
import config.classifier
from kernel_density_estimation.model import KDE #, select_KDE_model
import sys
import pandas as pd
import scipy.stats
import pickle


def true_pdf(x, _lambda, min_x, max_x):
    cdf_coef_pos = 1. / (1 - np.exp(- _lambda * max_x))
    cdf_coef_neg = 1. / (1 - np.exp(- _lambda * -min_x))
    cdf_coef = np.ones([x.size]) * cdf_coef_pos
    cdf_coef[x < 0] = cdf_coef_neg
    pdf = _lambda * np.exp(- _lambda * np.abs(x))
    return 0.5 * pdf * cdf_coef

if __name__ == '__main__':



    MAX_N = 2000
    _lambda = 0.5
    N1 = MAX_N // 2
    min_x = -30
    max_x = 30

    # samples_1 = np.random.exponential(1. / _lambda, N1)
    # samples_2 = -np.random.exponential(1. / _lambda, MAX_N - N1)
    # samples = np.concatenate([samples_1, samples_2])
    # np.random.shuffle(samples)
    # if np.max(samples) > max_x or np.min(samples) < min_x:
    #     raise Exception('sample generated out of range')
    # with open('twoSideExp_data.pyData', 'wb') as datafile:
    #     pickle.dump(samples, datafile)

    with open('twoSideExp_data.pyData', 'rb') as datafile:
        samples = pickle.load(datafile)
    n = 500
    test_samples = samples[1000:]
    samples = samples[:n]
    print('############', samples[-1])

    print(np.min(samples))
    print(np.max(samples))


    print(np.min(samples), np.max(samples))

    test_scaled_samples = scale_data(test_samples, min_x, max_x)
    scaled_samples = scale_data(samples, min_x, max_x)
    print(np.min(scaled_samples), np.max(scaled_samples))
    ticks = np.arange(min_x, max_x, (max_x - min_x) / 1000)

    avgLL_true = np.mean(np.log(true_pdf(test_samples, _lambda, min_x, max_x)))
    y_ticks_true = true_pdf(ticks,  _lambda, min_x, max_x)
    # plt.hist(samples, bins=100, density=True)
    plt.plot(ticks, y_ticks_true, color='blue')


    print('vkde:')
    # validation_threshold = n // 3
    best_kernel_width = None
    best_kde_avgLL = -np.Inf
    for kernel_width in np.arange(0.001, 0.1, 0.001):
        print('.', end='')
        sys.stdout.flush()
        # tmp_kde = KDE(scaled_samples[validation_threshold:], kernel_width)
        # tmp_avgLL = np.mean(tmp_kde.logpdf(scaled_samples[:validation_threshold]))
        tmp_kde = KDE(scaled_samples, kernel_width)
        tmp_avgLL = np.mean(tmp_kde.logpdf(test_scaled_samples))

        if tmp_avgLL > best_kde_avgLL:
            best_kde_avgLL = tmp_avgLL
            best_kernel_width = kernel_width
    best_kde = KDE(scaled_samples, best_kernel_width)

    # ## rule of thumb kernelwidth
    # best_kde = KDE(scaled_samples)
    # best_kernel_width = best_kde.kde.bandwidth
    # best_kde_avgLL = np.mean(best_kde.logpdf(test_scaled_samples))

    print(best_kernel_width)
    kde_avgll = np.mean(np.log(np.exp(best_kde.logpdf(test_scaled_samples)) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))))
    print('vkde KL: ', avgLL_true - kde_avgll)
    y_ticks_kde = np.exp(best_kde.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))

    plt.plot(ticks, y_ticks_kde, color='red')


    # best_k= None
    # best_logpoly_avgLL = -np.Inf
    # for k in [5,10,15,20]:
    #     logpoly = Logpoly()
    #     print('logpoly - k:', k)
    #     logpoly.fit(mp_compute_SS(scaled_samples[validation_threshold:], k), n-validation_threshold)
    #     tmp_avgLL = np.mean(logpoly.logpdf(scaled_samples[:validation_threshold]))
    #     if best_logpoly_avgLL < tmp_avgLL:
    #         best_logpoly_avgLL = tmp_avgLL
    #         best_k = k
    best_k = 20
    logpoly = Logpoly()
    print('logpoly - k:', best_k)
    logpoly.fit(mp_compute_SS(scaled_samples, best_k), n)
    logpoly_avgLL = np.mean(np.log(np.exp(logpoly.logpdf(test_scaled_samples)) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))))
    print('logpoly KL: ', avgLL_true - logpoly_avgLL)
    sys.stdout.flush()
    y_ticks_logpoly = np.exp(logpoly.logpdf(scale_data(ticks, min_x, max_x))) / (
                (max_x - min_x) / (1 - (config.logpoly.right_margin + config.logpoly.left_margin)))
    plt.plot(ticks, y_ticks_logpoly, color='green')

    # for k, c in [(20 , 'green')]: #[(5,'green'), (8,'green'), (11,'green'), (14,'green'), (17,'green')]:
    #     logpoly = Logpoly()
    #     print('logpoly - k:', k)
    #     logpoly.fit(mp_compute_SS(scaled_samples, k), n)
    #     # print('logpoly KL: ', avgLL_true - (logpoly.current_log_likelihood)/n)
    #     print('logpoly KL: ', avgLL_true - np.mean(logpoly.logpdf(test_scaled_samples)))
    #     sys.stdout.flush()
    #     y_ticks_logpoly = np.exp(logpoly.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))
    #     plt.plot(ticks, y_ticks_logpoly, color=c)

    plt.show()

    print('n = ', n)
    print('true:')
    head = 0
    while head < ticks.size:
        tail = head
        head = min(tail+100, ticks.size)
        for i in range(tail,  head):
            print('(' + str(ticks[i]) + ', ' + str(float(y_ticks_true[i])) + ') ', end='')
        print()
    print()

    print('logpoly:')
    head = 0
    while head < ticks.size:
        tail = head
        head = min(tail + 100, ticks.size)
        for i in range(tail, head):
            print('(' + str(ticks[i]) + ', ' + str(float(y_ticks_logpoly[i])) + ') ', end='')
        print()
    print()

    print('kde:')
    head = 0
    while head < ticks.size:
        tail = head
        head = min(tail + 100, ticks.size)
        for i in range(tail, head):
            print('(' + str(ticks[i]) + ', ' + str(float(y_ticks_kde[i])) + ') ', end='')
        print()
    print()

