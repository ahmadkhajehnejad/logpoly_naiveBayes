#from logpoly.model import Logpoly
#
#logpoly = Logpoly()
#logpoly.fit(plot=False)

from classifier.model import NaiveBayesClassifier, scorer
import config
import pandas as pd
import pickle as pk
import numpy as np
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sys

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

def k_fold_classify(data, labels, features_info):
    classifier = NaiveBayesClassifier(features_info)
    scores = cross_val_score(classifier, data, labels, scoring=scorer, cv=5)
    return scores

# def k_fold_avg_log_likelihood():
#     data, features_info = read_data()
#     np.random.shuffle(data)
#     kf = KFold(n_split=config.general.folds_count)
#     avg_log_likelihood_list = []
#     for train_index, test_index in kf.split(data):
#         nbc = NaiveBayesClassifier()
#         nbc.train(data[train_index, :], features_info)
#         avg_log_likelihood_list.append(nbc.get_avg_log_likelihood(data[test_index, :], features_info))
#     return np.mean(avg_log_likelihood_list)

if __name__ == '__main__':
    data, labels, features_info = load_data()
    scores = k_fold_classify(data, labels, features_info)
    print(scores)


