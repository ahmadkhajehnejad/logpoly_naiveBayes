from logpoly.tools import mp_compute_SS
import numpy as np
import config.logpoly
import config.classifier
import mpmath
import config.communication
from communication_tools import get_listener, receive_msg, send_msg, load_data
from multiprocessing import Process
import scipy.stats
import scipy.special

class DataClient:

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def get_data(self, dimension, class_):
        return self.data[self.labels == class_, dimension]


def make_data_clients():
    _, data_train, labels_train, _, _ = load_data()
    n = data_train.shape[0]
    sep = np.linspace(0, n, config.communication.num_clients + 1).astype(int)
    data_clients = []
    for i in range(config.communication.num_clients):
        data_clients.append(DataClient(data_train[sep[i]:sep[i + 1], :], labels_train[sep[i]:sep[i + 1]]))

    return data_clients


def _get_gmm_statistics(data, pi, mu, var):
    gamma = np.zeros([data.size, pi.size])
    for j in range(pi.size):
        gamma[:,j] = pi[j] * scipy.stats.norm.pdf(data, loc=mu[j], scale=np.sqrt(var[j]))
    gamma = gamma / np.tile(np.sum(gamma, axis=1, keepdims=True), [1, pi.size])
    sum_gamma = np.sum(gamma, axis=0)
    esum_x = np.matmul( data.reshape([1,-1]), gamma).reshape([-1])
    esum_x2 = np.matmul(data.reshape([1, -1])**2, gamma).reshape([-1])
    return [sum_gamma, esum_x, esum_x2]


def _get_gmm_avglogpdf(data, pi, mu, var):
    lp = np.zeros([data.size, pi.size])
    for j in range(pi.size):
        lp[:, j] = np.log(pi[j]) + scipy.stats.norm.logpdf(data, loc=mu[j], scale=np.sqrt(var[j]))
    lp = scipy.special.logsumexp(lp, axis=1)
    return np.sum(lp)

def thread_func(reply_address, data, msg):
    if msg[0] == 'get_n':
        result = data.shape[0]
        res_size = 1
    elif msg[0] == 'get_logpoly_SS':
        k, from_, to_ = msg[1], msg[2], msg[3]
        result = mp_compute_SS(data[from_:to_], k)
        res_size = result.size
    elif msg[0] == 'get_data':
        result = data
        res_size = data.size
    elif msg[0] == 'get_gmm_statistics':
        pi, mu, var, from_, to_= msg[1], msg[2], msg[3], msg[4], msg[5]
        result = _get_gmm_statistics(data[from_:to_], pi, mu, var)
        res_size = np.sum([s.size for s in result])
    elif msg[0] == 'get_gmm_avglogpdf':
        pi, mu, var, from_, to_ = msg[1], msg[2], msg[3], msg[4], msg[5]
        result = _get_gmm_avglogpdf(data[from_:to_], pi, mu, var)
        res_size = 1
    send_msg(reply_address, result, res_size)


if __name__ == '__main__':

    listener = get_listener()
    with open('config/client_nodes_address.py', 'w') as file:
        file.write('client_nodes_address =' + str(listener.address) + '\n')

    mpmath.mp.dps = config.logpoly.mp_dps
    data_clients = make_data_clients()


    while True:
        msg = receive_msg(listener)
        if msg[0] == 'close':
            break
        else:
            reply_address = msg[0]
            data_client_num = msg[1]
            dimension = msg[2]
            class_ = msg[3]
            Process(target=thread_func,
                    args=[reply_address, data_clients[data_client_num].get_data(dimension, class_), msg[4:]],
                    daemon=True).start()

    listener.close()
