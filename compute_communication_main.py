
from classifier.center_model import NaiveBayesClassifier, scorer
import config.general
import config.classifier
import pandas as pd
import numpy as np
import sys
import logpoly
from multiprocessing import Process
from data_client import run_data_client
from multiprocessing.connection import Client


MAX_DIMENSION = 114
MAX_TRAIN_SIZE = 10000

def make_clients_and_load_test_data():
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

    data = data[:, :MAX_DIMENSION]  ########################
    features_info = features_info[:MAX_DIMENSION] + [features_info[-1]] ####################

    scaled_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        scaled_data[:, i] = logpoly.tools.scale_data(data[:, i], features_info[i]['min_value'],
                                                     features_info[i]['max_value'])

    n_test = config.classifier.test_size
    scaled_data_test = scaled_data[:n_test, :]
    labels_test = labels[:n_test]

    n = scaled_data.shape[0]
    n = config.classifier.test_size + MAX_TRAIN_SIZE ###################
    sep = np.linspace(n_test, n+1, config.general.num_clients+1).astype(int)

    client_processes = []

    for i in range(config.general.num_clients):
        client_processes.append(
            Process(target=run_data_client, args=[i, scaled_data[sep[i]:sep[i + 1], :], labels[sep[i]:sep[i + 1]]],
                    daemon=True))
        client_processes[i].start()

    return client_processes, scaled_data_test, labels_test, features_info



if __name__ == '__main__':
    client_processes, data_test, labels_test, features_info = make_clients_and_load_test_data()
    classifier = NaiveBayesClassifier(features_info)
    classifier.fit()

    for i in range(config.general.num_clients):

        address = ('localhost', config.general.first_client_port + i)  # family is deduced to be 'AF_INET'
        conn = Client(address)  # , authkey='secret password')
        conn.send(['close'])
        conn.close()

        client_processes[i].join()
        client_processes[i].terminate()

    score = scorer(classifier, data_test, labels_test)
    print('\ntest score:  ', score)


