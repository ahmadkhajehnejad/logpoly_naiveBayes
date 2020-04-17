import numpy as np
from config.communication import PORT_RANGE
from multiprocessing.connection import Listener, Client
import socket
import pandas as pd
import config.general
import config.communication
import logpoly

MAX_DIMENSION = config.communication.MAX_DIMENSION
MAX_TRAIN_SIZE = config.communication.MAX_TRAIN_SIZE


def get_listener():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    my_ip = s.getsockname()[0]
    s.close()

    found = False
    while not found:
        found = True
        try:
            my_port = np.random.randint(PORT_RANGE[0], PORT_RANGE[1])
            address = (my_ip, my_port)  # family is deduced to be 'AF_INET'
            listener = Listener(address)  # , authkey='secret password')
        except:
            found = False
    return listener

def receive_msg(listener):
    conn = listener.accept()
    src = listener.last_accepted
    trg = listener.address
    print('## from_port_to_port:', src[0], ':', src[1],  '->', trg[0], ':', trg[1])
    msg = conn.recv()
    conn.close()
    return msg

def send_msg(address, msg, msg_size):
    connected = False
    while not connected:
        try:
            connected = True
            conn = Client(address)  # , authkey='secret password')
        except:
            connected = False
    print('@@ sending ' + str(msg_size) + ' variables.')
    conn.send(msg)
    conn.close()


def load_data():
    data = pd.read_csv('data/' + config.general.dataset_name + '.csv', header=None).values
    labels = data[:, -1]
    data = data[:, :-1]
    print('data shape:', data.shape) 

    feature_types = pd.read_csv('data/' + config.general.dataset_name + '_feature_types.csv',
                                header=None).values.reshape([-1])
    feature_types = np.array([f.strip() for f in feature_types])
    features_info = []
    for i in range(len(feature_types)):
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

    data = data[:, :MAX_DIMENSION]  ########################
    features_info = features_info[:MAX_DIMENSION] + [features_info[-1]]  ####################

    scaled_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        scaled_data[:, i] = logpoly.tools.scale_data(data[:, i], features_info[i]['min_value'],
                                                     features_info[i]['max_value'])

    n_test = config.classifier.test_size

    # n = scaled_data.shape[0]
    n = config.classifier.test_size + MAX_TRAIN_SIZE  ###################

    scaled_test_data = scaled_data[:n_test, :]
    test_labels = labels[:n_test]

    scaled_train_data = scaled_data[n_test:n, :]
    train_labels = labels[n_test:]

    return features_info, scaled_train_data, train_labels, scaled_test_data, test_labels
