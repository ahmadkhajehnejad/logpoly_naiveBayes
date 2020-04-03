
from classifier.center_model import NaiveBayesClassifier, scorer
import config
import pandas as pd
import numpy as np
import sys
from client import Client


def get_clients_and_test_data(features_info):
    data = pd.read_csv('data/' + config.general.dataset_name + '.csv', header = None).values
    labels = data[:, -1]
    data = data[:, :-1]
    n = data.shape[0]

    n_test = n // config.classifier.test_size
    data_test = data[:n_test, :]
    labels_test = labels[:n_test]

    data = data[n_test:, :]
    labels = labels[n_test:]
    n = data.shape[0]

    ind = np.random.permutation(n)
    data = data[ind,:]
    labels = labels[ind]

    partition_points = np.round(np.linspace(0, n, config.general.num_clients+1)).astype(int)
    clients = []
    for i in range(config.general.num_clients):
        selected_ind = ind[ partition_points[i]: partition_points[i+1]]
        clients.append(Client(data[ selected_ind, :], labels[selected_ind], features_info))

    return clients, data_test, labels_test


def get_features_info():
    data = pd.read_csv('data/' + config.general.dataset_name + '.csv', header=None).values
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
    return features_info


if __name__ == '__main__':
    features_info = get_features_info()
    clients, data_test, labels_test = get_clients_and_test_data(features_info)
    classifier = NaiveBayesClassifier(features_info, clients)
    classifier.fit()

    for i, c in enumerate(clients):
        print('client #' + str(i), '   sent bytes: ', c.sent_bytes, '       communication_rounds: ', c.communication_rounds)


    score = scorer(NaiveBayesClassifier, data_test, labels_test)

    print('sum of sent bytes: ', np.sum([c.sent_bytes for c in clients]))
    print('sent bytes by each client: ', [c.sent_bytes for c in clients])
    print('\nsum of received bytes: ', np.sum([c.received_bytes for c in clients]))
    print('received bytes by each client: ', [c.received_bytes for c in clients])
    print('\nsum of communication rounds: ', np.sum([c.communication_rounds for c in clients]))
    print('communication rounds by each client: ', [c.communication_rounds for c in clients])

    print('\ntest score:  ', score)

    #classifier.test()

    # print(scores)
    # print(np.mean(scores))
    # print(np.std(scores))

