import numpy as np
# import matplotlib.pyplot as plt
from logpoly.model import Logpoly, _compute_objective_function, LogpolyModelSelector
from logpoly.tools import mp_compute_SS, scale_data
import config.logpoly
from Gaussian_mixture_model.model import GaussianMixtureModel
import sys
import pandas as pd
import pickle as pkl
from multiprocessing import Process, Queue


def load_data(device, feature_num, cls):
    file_path = 'data/iot_botnet_attack_' + device + '_11classes.csv'
    data = pd.read_csv(file_path, header=None).sample(frac=1).reset_index(drop=True).values
    y = data[:, -1]
    data_feature = data[:, feature_num]
    return data_feature[y == cls]


def fit_thread_func(shared_space, SS_train, n, scaled_samples_validation, scaled_samples_test):
    k = SS_train.size - 1
    logpoly = Logpoly()
    print('logpoly - k:', k)
    logpoly.fit(SS_train[:k + 1], n)

    avgLL_train = (logpoly.current_log_likelihood) / n
    avgLL_validation = np.mean(logpoly.logpdf(scaled_samples_validation))
    avgLL_test = np.mean(logpoly.logpdf(scaled_samples_test))

    shared_space.put([k, avgLL_train, avgLL_validation, avgLL_test, logpoly.theta, logpoly.logZ])

if __name__ == '__main__':

    device = 'Danmini_Doorbell'
    samples = load_data(device, 0, 1)
    n_test = 500
    n_val = 500
    n_train = 3000
    n = n_test + n_val + n_train
    samples = samples[:n]
    print(n)

    print(np.min(samples), np.max(samples))
    min_x = np.min(samples)
    max_x = np.max(samples)

    scaled_samples = scale_data(samples, min_x, max_x)
    print(np.min(scaled_samples), np.max(scaled_samples))
    ticks = np.arange(min_x, max_x, 0.01)

    # plt.hist(samples, bins=2000, density=True)

    MAX_k = 20

    scaled_samples_validation = scaled_samples[:n_val]
    # SS_validation = mp_compute_SS(scaled_samples_validation, MAX_k)

    scaled_samples_test = scaled_samples[n_val:n_val+n_test]
    # SS_test = mp_compute_SS(scaled_samples_test, MAX_k)

    scaled_samples_train = scaled_samples[n_test+n_val:]
    SS_train = mp_compute_SS(scaled_samples_train, MAX_k)




    avgLL_list_train = np.zeros([MAX_k])
    avgLL_list_validation = np.zeros([MAX_k])
    avgLL_list_test = np.zeros([MAX_k])
    theta_list = [None] * MAX_k
    logZ_list = [None] * MAX_k

    shared_space = Queue()
    processes = []

    for k in range(1,MAX_k+1):
        processes.append(Process(target=fit_thread_func,
                                 args=[shared_space, SS_train[:k+1], n_train, scaled_samples_validation, scaled_samples_test], daemon=True))

    max_num_processes = 30
    head = np.min([max_num_processes, len(processes)])
    for i in range(head):
        processes[i].start()

    num_terminated = 0
    while num_terminated < len(processes):
        res = shared_space.get()
        k = res[0]
        avgLL_train = res[1]
        avgLL_validation = res[2]
        avgLL_test = res[3]
        theta = res[4]
        logZ = res[5]

        avgLL_list_train[k - 1] = avgLL_train
        avgLL_list_validation[k - 1] = avgLL_validation
        avgLL_list_test[k - 1] = avgLL_test
        theta_list[k - 1] = theta
        logZ_list[k - 1] = logZ

        print('k:', k, '   avgLL_train: ', avgLL_train, '  avgLL_val: ', avgLL_validation, '  avgLL_test: ', avgLL_test)
        sys.stdout.flush()

        processes[k - 1].join()
        processes[k - 1].terminate()
        processes[k - 1] = None
        num_terminated += 1

        if head < len(processes):
            processes[head].start()
            head += 1

    shared_space.close()




    # y_ticks_logpoly = np.exp(logpoly.logpdf(scale_data(ticks, min_x, max_x))) / (
    #             (max_x - min_x) / (1 - (config.logpoly.right_margin + config.logpoly.left_margin)))
    # plt.plot(ticks, y_ticks_logpoly)
    # plt.show()

    with open('submodel_selection.pydata', 'wb') as outfile:
        pkl.dump([avgLL_list_train, avgLL_list_validation, avgLL_list_test, theta_list, logZ_list], outfile)

    print('train:' , ['('+str(i+1)+','+str(a)+')' for i,a in enumerate(avgLL_list_train)])
    print('validation:', ['('+str(i+1)+','+str(a)+')' for i,a in enumerate(avgLL_list_validation)])
    print('test', ['('+str(i+1)+','+str(a)+')' for i,a in enumerate(avgLL_list_test)])
