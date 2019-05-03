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


def true_pdf(x, min_D, ll, rr, max_D, p1, p2):
    result = np.zeros([x.size])
    result[ (x >= min_D) & (x < ll) ] = p1
    result[ (x >= ll) & (x <= rr) ] = p2
    result[ (x > rr) & (x <= max_D) ] = p1
    return result

if __name__ == '__main__':



    MAX_N = 2000
    min_D = -0.9
    max_D = 0.9
    ll = -0.2
    rr =  0.2
    p1 = 0.2
    p2 = p1 + ((1 - (p1 * (max_D - min_D))) / (rr - ll))
    print(p1, p2)
    N1 = int(p1 * (max_D - min_D) * MAX_N)
    print(N1, MAX_N-N1)

    # samples_1 = np.random.uniform(min_D, max_D, N1)
    # samples_2 = np.random.uniform(ll, rr, MAX_N - N1)
    # samples = np.concatenate([samples_1, samples_2])
    # np.random.shuffle(samples)
    # with open('pulse_data.pyData', 'wb') as datafile:
    #     pickle.dump(samples, datafile)
    # test_samples = samples[10000:]

    with open('pulse_data.pyData', 'rb') as datafile:
        samples = pickle.load(datafile)
    n = 750
    test_samples = samples[1000:]
    samples = samples[:n]
    print('############', samples[-1])

    print(np.min(samples))
    print(np.max(samples))
    min_x = -1
    max_x = 1

    print(np.min(samples), np.max(samples))

    test_scaled_samples = scale_data(test_samples, min_x, max_x)
    scaled_samples = scale_data(samples, min_x, max_x)
    print(np.min(scaled_samples), np.max(scaled_samples))
    ticks = np.arange(min_x, max_x, (max_x - min_x) / 1000)

    avgLL_true = np.mean(true_pdf(test_samples, min_D, ll, rr, max_D, p1, p2))
    y_ticks_true = true_pdf(ticks, min_D, ll, rr, max_D, p1, p2)
    plt.hist(samples, bins=1000, density=True)
    plt.plot([min_x, min_D, min_D, ll, ll, rr, rr, max_D, max_D, max_x], [0, 0, p1, p1, p2, p2, p1, p1, 0, 0], color='blue')

    print('vkde:')
    validation_threshold = n // config.classifier.validation_portion
    best_kernel_width = None
    best_kde_avgLL = -np.Inf
    for kernel_width in np.arange(0.001, 0.1, 0.001):
        print('.', end='')
        sys.stdout.flush()
        tmp_kde = KDE(scaled_samples[validation_threshold:], kernel_width)
        tmp_avgLL = np.mean(tmp_kde.logpdf(scaled_samples[:validation_threshold]))
        if tmp_avgLL > best_kde_avgLL:
            best_kde_avgLL = tmp_avgLL
            best_kernel_width = kernel_width
    best_kde = KDE(scaled_samples, best_kernel_width)

    # ## rule of thumb kernelwidth
    # best_kde = KDE(scaled_samples)
    # best_kernel_width = best_kde.kde.bandwidth
    # best_kde_avgLL = np.mean(best_kde.logpdf(test_scaled_samples))

    print(best_kernel_width)
    print('vkde KL: ', avgLL_true - np.mean(best_kde.logpdf(test_scaled_samples)))
    y_ticks_kde = np.exp(best_kde.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))

    plt.plot(ticks, y_ticks_kde, color='red')

    for k, c in [(20 , 'green')]: #[(5,'green'), (8,'green'), (11,'green'), (14,'green'), (17,'green')]:
        logpoly = Logpoly()
        print('logpoly - k:', k)
        logpoly.fit(mp_compute_SS(scaled_samples, k), n)
        # print('logpoly KL: ', avgLL_true - (logpoly.current_log_likelihood)/n)
        print('logpoly KL: ', avgLL_true - np.mean(logpoly.logpdf(test_scaled_samples)))
        sys.stdout.flush()
        y_ticks_logpoly = np.exp(logpoly.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))
        plt.plot(ticks, y_ticks_logpoly, color=c)

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

