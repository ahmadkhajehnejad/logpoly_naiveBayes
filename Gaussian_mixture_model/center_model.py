import numpy as np
import config.gmm
from communication_tools import get_listener, receive_msg, send_msg
from config.client_nodes_address import client_nodes_address
import config.logpoly
import config.communication
import config.classifier


class DistributedGaussianMixtureModel:

    def __init__(self, listener, dimension, class_, num_components, from_, to_):

        self.mu = np.random.random(num_components) * (config.logpoly.x_ubound - config.logpoly.x_lbound) + config.logpoly.x_lbound
        self.var = np.random.rand(num_components) * (config.logpoly.x_ubound - config.logpoly.x_lbound) # / (2 * num_components)
        self.pi = np.ones([num_components]) / num_components

        n = np.sum([to_[i] - from_[i] for i in range(config.communication.num_clients)])

        for iter in range(config.gmm.max_iter):
            # print('        iter', iter)
            sum_gamma = np.zeros([num_components])
            esum_x = np.zeros([num_components])
            esum_x2 = np.zeros([num_components])
            for client_number in range(config.communication.num_clients):
                msg_sz = 8
                send_msg(client_nodes_address, [listener.address, client_number, dimension, class_, 'get_gmm_statistics',
                                                self.pi, self.mu, self.var, from_[client_number], to_[client_number]], msg_sz)
                msg = receive_msg(listener)
                sum_gamma += msg[0]
                esum_x += msg[1]
                esum_x2 += msg[2]
            self.mu = esum_x / sum_gamma
            self.var = (esum_x2 + sum_gamma * self.mu * self.mu - 2 * esum_x * self.mu) / sum_gamma
            min_allowable_var = 0.000001
            self.var[self.var < min_allowable_var] = min_allowable_var
            self.pi = sum_gamma / n

    def avglogpdf(self, listener, dimension, class_, from_, to_):
        result = 0
        for client_number in range(config.communication.num_clients):
            msg_sz = 8
            send_msg(client_nodes_address, [listener.address, client_number, dimension, class_, 'get_gmm_avglogpdf',
                                            self.pi, self.mu, self.var, from_[client_number], to_[client_number]], msg_sz)
            msg = receive_msg(listener)
            result += msg
        return result

    def logpdf(self, x):
        return np.array(self.gmm.score_samples(np.array(x).reshape([-1, 1])))


def select_GMM_model( dimension, class_, list_of_num_components):

    listener = get_listener()

    def _get_client_n(client_number):
        msg_sz = 3
        send_msg(client_nodes_address, [listener.address, client_number, dimension, class_, 'get_n'], msg_sz)
        msg = receive_msg(listener)
        return msg

    n_clients = [_get_client_n(i) for i in range(config.communication.num_clients)]
    n_validation_clients = [ n // config.classifier.validation_portion for n in n_clients ]

    if len(list_of_num_components) == 1:
        result = DistributedGaussianMixtureModel(listener, dimension, class_, num_components=list_of_num_components[0],
                                                 from_=[0 for _ in n_clients], to_=n_clients)
        listener.close()
        return result

    # if config.classifier.smart_validation:
    #     ind = np.argsort(data)
    # else:
    #     ind = np.arange(n_total)
    # index_train, index_validation = get_train_and_validation_index(ind)
    # n_train = index_train.size

    gmm_models = []
    avg_log_likelihoods = []
    for i, k in enumerate(list_of_num_components):
        # print('   ', k, 'components')
        gmm_models.append(DistributedGaussianMixtureModel(listener, dimension, class_, num_components=k,
                                                          from_=n_validation_clients, to_=n_clients))
        validation_score = gmm_models[i].avglogpdf(listener, dimension, class_, from_=[0 for _ in n_clients], to_=n_validation_clients)
        avg_log_likelihoods.append(np.mean(validation_score))

    best_index = np.argmax(avg_log_likelihoods)
    result = DistributedGaussianMixtureModel(listener, dimension, class_, num_components=list_of_num_components[best_index],
                                             from_=[0 for _ in n_clients], to_=n_clients)
    listener.close()
    return result
