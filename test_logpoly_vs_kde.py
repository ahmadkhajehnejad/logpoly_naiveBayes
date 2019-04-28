import numpy as np
import matplotlib.pyplot as plt
from logpoly.model import Logpoly, _compute_log_likelihood, LogpolyModelSelector
from logpoly.tools import mp_compute_SS, scale_data
import config.logpoly
from kernel_density_estimation.model import  select_KDE_model
import sys
import pandas as pd
import scipy.stats



if __name__ == '__main__':



    n = 300
    min_D = -0.9
    max_D = 0.9
    ll = -0.2
    rr =  0.2
    p1 = 0.2
    p2 = p1 + ((1 - (p1 * (max_D - min_D))) / (rr - ll))
    print(p1, p2)
    n1 = int(p1 * (max_D - min_D) * n)
    print(n1, n-n1)
    samples_1 = np.random.uniform(min_D, max_D, n1)
    samples_2 = np.random.uniform(ll, rr, n - n1)
    samples = np.concatenate([samples_1, samples_2])
    np.random.shuffle(samples)
    print(np.min(samples))
    print(np.max(samples))
    min_x = -1
    max_x = 1

    print(np.min(samples), np.max(samples))

    scaled_samples = scale_data(samples, min_x, max_x)
    print(np.min(scaled_samples), np.max(scaled_samples))
    ticks = np.arange(min_x, max_x, (max_x - min_x) / 1000)

    # avgLL_true = np.mean(triangular_pdf(samples, ll, md, rr))

    plt.plot([min_x, min_D, min_D, ll, ll, rr, rr, max_D, max_D, max_x], [0, 0, p1, p1, p2, p2, p1, p1, 0, 0], color='blue')

    print('vkde:')
    kde = select_KDE_model(scaled_samples, np.arange(0.001, 0.5, 0.001))
    y_ticks_kde = np.exp(kde.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))

    plt.plot(ticks, y_ticks_kde, color='red')

    for k, c in [(20 , 'green')]: #[(5,'green'), (8,'green'), (11,'green'), (14,'green'), (17,'green')]:
        logpoly = Logpoly()
        print('logpoly - k:', k)
        logpoly.fit(mp_compute_SS(scaled_samples, k), n)
        # print('logpoly KL: ', avgLL_true - (logpoly.current_log_likelihood)/n)
        sys.stdout.flush()
        y_ticks_logpoly = np.exp(logpoly.logpdf(scale_data(ticks, min_x, max_x))) / ((max_x - min_x) / (1-(config.logpoly.right_margin + config.logpoly.left_margin)))
        plt.plot(ticks, y_ticks_logpoly, color=c)

    plt.show()

    # print('true:')
    # head = 0
    # while head < ticks.size:
    #     tail = head
    #     head = min(tail+100, ticks.size)
    #     for i in range(tail,  head):
    #         print('(' + str(ticks[i]) + ', ' + str(float(y_ticks_true[i])) + ') ', end='')
    #     print()
    # print()
    #
    # print('logpoly:')
    # head = 0
    # while head < ticks.size:
    #     tail = head
    #     head = min(tail + 100, ticks.size)
    #     for i in range(tail, head):
    #         print('(' + str(ticks[i]) + ', ' + str(float(y_ticks_logpoly[i])) + ') ', end='')
    #     print()
    # print()
    #
    # print('gmm:')
    # head = 0
    # while head < ticks.size:
    #     tail = head
    #     head = min(tail + 100, ticks.size)
    #     for i in range(tail, head):
    #         print('(' + str(ticks[i]) + ', ' + str(float(y_ticks_gmm[i])) + ') ', end='')
    #     print()
    # print()
    #
