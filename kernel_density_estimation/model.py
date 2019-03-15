from sklearn.neighbors.kde import KernelDensity
import numpy as np
import config.classifier
from tools import get_train_and_validation_index


class KDE:

    def __init__(self, data, bandwidth = None):
        if bandwidth is None:
            bandwidth = 1. / np.sqrt(data.size)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(data).reshape([-1, 1]))


    def logpdf(self, x):
        return np.array(self.kde.score_samples(np.array(x).reshape([-1, 1])))


def select_KDE_model(data, list_of_bandwidths):

    n_total = data.shape[0]

    if config.classifier.smart_validation:
        ind = np.argsort(data)
    else:
        ind = np.arange(n_total)
    index_train, index_validation = get_train_and_validation_index(ind)
    n_train = index_train.size

    kde_models = []
    avg_log_likelihoods = []
    for i, w in enumerate(list_of_bandwidths):
        kde_models.append(KDE(data[index_train], bandwidth=w))
        avg_log_likelihoods.append(np.mean(kde_models[i].logpdf(data[index_validation])))

    best_index = np.argmax(avg_log_likelihoods)
    return KDE(data, bandwidth=list_of_bandwidths[best_index])
