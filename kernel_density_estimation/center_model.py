from sklearn.neighbors.kde import KernelDensity
import numpy as np
import config.classifier
import config.communication
from tools import get_train_and_validation_index
from config.client_nodes_address import client_nodes_address
from communication_tools import get_listener, receive_msg, send_msg


def _get_data_from_clients(dimension, class_):
    listener = get_listener()

    data_from_clients = []
    for client_number in range(config.communication.num_clients):
        send_msg(client_nodes_address, [listener.address, client_number, dimension, class_, 'get_data'])
        msg = receive_msg(listener)
        data_from_clients.append(msg)

    listener.close()

    data = np.concatenate(data_from_clients, axis=0)
    return data

class KDE:

    def __init__(self, dimension, class_, bandwidth = None):
        data = _get_data_from_clients(dimension, class_)

        if bandwidth is None:
            bandwidth = 1. / np.sqrt(data.size)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(data).reshape([-1, 1]))


    def logpdf(self, x):
        return np.array(self.kde.score_samples(np.array(x).reshape([-1, 1])))


def select_KDE_model(dimension, class_, list_of_bandwidths):

    data = _get_data_from_clients(dimension, class_)

    n_total = data.shape[0]

    if config.classifier.smart_validation:
        ind = np.argsort(data)
    else:
        ind = np.arange(n_total)
    index_train, index_validation = get_train_and_validation_index(ind)
    n_train = index_train.size

    if isinstance(list_of_bandwidths, str) and list_of_bandwidths == "around_heuristic":
        b = 1. / np.sqrt(n_train)
        list_of_bandwidths = [b / 2, b, 2 * b]


    kde_models = []
    avg_log_likelihoods = []
    for i, w in enumerate(list_of_bandwidths):
        kde_models.append(KDE(data[index_train], bandwidth=w))
        avg_log_likelihoods.append(np.mean(kde_models[i].logpdf(data[index_validation])))

    best_index = np.argmax(avg_log_likelihoods)
    # print('best kernel width:', list_of_bandwidths[best_index])
    return KDE(data, bandwidth=list_of_bandwidths[best_index])
