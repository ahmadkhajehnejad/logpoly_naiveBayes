from logpoly.tools import  mp_compute_poly, mp_compute_SS
import logpoly
import numpy as np
import sys
import pandas as pd
import argparse
import config
import mpmath
from multiprocessing.connection import Listener
from multiprocessing.connection import Client

def load_data():
    data = pd.read_csv('data/' + config.general.dataset_name + '.csv', header = None).values
    labels = data[:, -1]
    data = data[:, :-1]
    feature_types = pd.read_csv('data/' + config.general.dataset_name + '_feature_types.csv',
                                header=None).values.reshape([-1])
    feature_types = np.array([f.strip() for f in feature_types])
    features_info = []
    for i in range(len(feature_types)):
        sys.stdout.flush()
        if feature_types[i] == config.general.CONTINUOUS_FEATURE:
            features_info.append({'feature_type': config.general.CONTINUOUS_FEATURE, 'min_value': np.min(data[:, i]),
                                  'max_value': np.max(data[:, i])})
        elif feature_types[i] == config.general.CATEGORICAL_FEATURE:
            features_info.append(
                {'feature_type': config.general.CATEGORICAL_FEATURE, 'categories': np.unique(data[:, i])})
        else:
            raise Exception('not handled case feature_type=' + str(feature_types[i]) + \
                  ' (dimension #' + str(i) + ')')
    features_info.append({'classes': np.unique(labels)})
    return [data, labels, features_info]


class Client:

    def __init__(self, client_number):
        data, labels, features_info = load_data()

        n_total = data.shape[0] - config.classifier.test_size

        n = n_total // config.general.num_clients

        start_ind = config.classifier.test_size + n * client_number
        end_ind = start_ind + n

        self.data = data[start_ind:end_ind,:]
        self.label = labels[start_ind:end_ind]

        self.scaled_data = np.zeros(self.data.shape)
        for i in range(data.shape[1]):
            self.scaled_data[:,i] = logpoly.tools.scale_data(self.data[:,i], features_info[i]['min_value'], features_info[i]['max_value'])


    def get_n(self):
        return self.scaled_data.shape[0]


    def get_logpoly_SS(self, d, c, k, from_, to_):
        ax = np.arange(self.data.shape[0])
        ind = np.where( (self.labels == c) & (ax >= from_) & (ax < to_))[0]
        return mp_compute_SS(self.scaled_data[ind,d], k)

    def get_data(self, dimension, class_):
        return self.data[self.labels == class_, dimension]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--client_number', help='The client\'s number.', action='store',
                        type=int)
    args = parser.parse_args()

    client_number = args.client_number

    mpmath.mp.dps = config.logpoly.mp_dps

    client = Client(client_number)

    address = ('localhost', 3344 + client_number)  # family is deduced to be 'AF_INET'
    listener = Listener(address)  # , authkey='secret password')

    while True:

        conn = listener.accept()
        msg = conn.recv()
        conn.close()

        if msg[0] == 'close':
            break

        reply_port = msg[0]
        if msg[1] == 'get_n':
            result = client.get_n()
        elif msg[1] == 'get_logpoly_SS':
            result = client.get_logpoly_SS(msg[2], msg[3], msg[4], msg[5], msg[6])
        elif msg[1] == 'get_data':
            result = client.get_data(msg[2],msg[3])

        address = ('localhost', reply_port)
        conn = Client(address)  # , authkey='secret password')
        conn.send(result)
        conn.close()

    listener.close()