
from classifier.center_model import NaiveBayesClassifier, scorer
import config
import pandas as pd
import numpy as np
import sys
from client import Client


def load_test_data():
    data = pd.read_csv('data/' + config.general.dataset_name + '.csv', header = None).values
    labels = data[:, -1]
    data = data[:, :-1]
    n_test = config.classifier.test_size
    data_test = data[:n_test, :]
    labels_test = labels[:n_test]

    return data_test, labels_test


def load_features_info():
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
    features_info = load_features_info()
    data_test, labels_test = load_test_data(features_info)
    classifier = NaiveBayesClassifier(features_info)
    classifier.fit()

    score = scorer(classifier, data_test, labels_test)

    print('\ntest score:  ', score)

    #classifier.test()

    # print(scores)
    # print(np.mean(scores))
    # print(np.std(scores))

