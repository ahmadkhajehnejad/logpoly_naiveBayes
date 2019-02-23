from sklearn.neighbors.kde import KernelDensity
import numpy as np


class KDE:

    def __init__(self, data, bandwidth = None):
        if bandwidth is None:
            bandwidth = 1. / np.sqrt(data.size)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(data).reshape([-1, 1]))


    def logpdf(self, x):
        return np.array(self.kde.score_samples(np.array(x).reshape([-1, 1])))


def select_KDE_model(data, list_of_bandwidths):

    n_total = data.shape[0]

    index_validation = np.arange(3, n_total, 4)
    index_tmp = np.ones([n_total])
    index_tmp[index_validation] = 0
    index_train = np.where(index_tmp)[0]

    # ind = np.arange(n_total)
    # np.random.shuffle(ind)
    # n_train = n_total // 4
    # index_train = ind[:n_train]
    # index_validation = ind[n_train:]




    kde_models = []
    avg_log_likelihoods = []
    for i, w in enumerate(list_of_bandwidths):
        kde_models.append(KDE(data[index_train], bandwidth=w))
        avg_log_likelihoods.append(np.mean(kde_models[i].logpdf(data[index_validation])))

    # print(list_of_bandwidths[np.argmax(avg_log_likelihoods)])
    return kde_models[np.argmax(avg_log_likelihoods)]
