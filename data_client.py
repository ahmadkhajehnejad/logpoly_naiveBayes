from logpoly.tools import  mp_compute_poly, mp_compute_SS
import logpoly
import numpy as np
import sys
import pandas as pd
import argparse
import config.general
import config.logpoly
import config.classifier
import mpmath
from multiprocessing.connection import Listener
from multiprocessing.connection import Client

def load_data():
    data = pd.read_csv('data/' + config.general.dataset_name + '.csv', header = None).values
    labels = data[:, -1]
    data = data[:, :config.general.max_dimension]  # :-1] ##################
    feature_types = pd.read_csv('data/' + config.general.dataset_name + '_feature_types.csv',
                                header=None).values.reshape([-1])
    feature_types = np.array([f.strip() for f in feature_types])
    feature_types = feature_types[:config.general.max_dimension]  ######################
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


class DataClient:

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def get_n(self):
        return self.scaled_data.shape[0]


    def get_logpoly_SS(self, d, c, k, from_, to_):
        ax = np.arange(self.data.shape[0])
        ind = np.where( (self.labels == c) & (ax >= from_) & (ax < to_))[0]
        return mp_compute_SS(self.scaled_data[ind,d], k)

    def get_data(self, dimension, class_):
        return self.data[self.labels == class_, dimension]


def run_data_client(client_number, data, labels):

    mpmath.mp.dps = config.logpoly.mp_dps

    data_client = DataClient(data, labels)
    my_port = config.general.first_client_port + client_number
    address = ('localhost', my_port)  # family is deduced to be 'AF_INET'
    listener = Listener(address)  # , authkey='secret password')

    while True:

        conn = listener.accept()
        print('## from_port_to_port: ' + str(listener.last_accepted) + ' -> ' + str(my_port))
        msg = conn.recv()
        conn.close()

        if msg[0] == 'close':
            break

        reply_port = msg[0]
        if msg[1] == 'get_n':
            result = data_client.get_n()
        elif msg[1] == 'get_logpoly_SS':
            result = data_client.get_logpoly_SS(msg[2], msg[3], msg[4], msg[5], msg[6])
        elif msg[1] == 'get_data':
            result = data_client.get_data(msg[2], msg[3])

        address = ('localhost', reply_port)
        conn = Client(address)  # , authkey='secret password')
        conn.send(result)
        conn.close()

    listener.close()
    print('client ' + str(client_number) + ' closed')
