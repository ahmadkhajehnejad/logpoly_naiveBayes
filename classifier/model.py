
import logpoly
from logpoly.model import LogpolyModelSelector
from categorical.model import CategoricalDensityEstimator
import config
import numpy as np



class NaiveBayesClassifier:

    def __init__(self, features_info):
        self.features_info = features_info
        self.density_estimators = None
        self.classes = self.features_info[-1]['classes']

    def fit(self, data, labels):
        self.density_estimators = [None] * len(self.classes)
        for c in self.classes:
            class_index = labels == c
            class_data = data[class_index, :]
            for i in range(len(self.features_info) - 1):
                if self.features_info[i]['feature_type'] == config.general.CONTINUOUS_FEATURE:
                    logpoly_model_selector = LogpolyModelSelector(config.logpoly.list_factor_degrees)
                    scaled_data = logpoly.tools.scale_data(class_data[:, i], self.features_info[i]['min_value'],
                                                           self.features_info[i]['max_value'])
                    self.density_estimators.append(logpoly_model_selector.select_model(scaled_data))
                elif self.features_info[i]['feature_type'] == config.general.CATEGORICAL_FEATURE:
                    self.density_estimators.append(
                        CategoricalDensityEstimator(class_data[:, i], self.features_info[i]['categories']))
            else:
                raise 'not handled case feature_type=' + str(self.features_info[i]['feature_type']) + \
                      ' (dimension #' + str(i) + ')'

    def get_log_likelihood_per_class(self, data):
        log_likelihood_per_class = np.zeros([data.shape[0], len(self.classes)])
        for c_index, c in enumerate(self.classses):
            log_likelihood_per_dim = np.zeros([data.shape[0], len(self.features_info) - 1])
            for i in range(len(self.features_info) - 1):
                if self.features_info[i]['feature_type'] == config.general.CONTINUOUS_FEATURE:
                    scaled_data = logpoly.tools.scale_data(data[:, i], self.features_info[i]['min_value'],
                                                           self.features_info[i]['max_value'])
                    log_likelihood_per_dim[:, i] = self.density_estimators[i].logpdf(scaled_data)
                else:
                    log_likelihood_per_dim[:, i] = self.density_estimators[i].logpdf(data[:, i])
            log_likelihood_per_class[:, c_index] = np.sum(log_likelihood_per_dim, axis=1)
        return log_likelihood_per_class

    def label(self, data):
        log_likelihood_per_class = self.get_log_likelihood_per_class(data)
        predicted_labels = np.argmax(log_likelihood_per_class, axis=1)
        return self.classes[predicted_labels]

    def get_params(self, deep=False):
        return {'features_info': self.features_info}


def scorer(classifier, data, labels):
    predicted_labels = classifier.label(data)
    score = np.sum(labels == predicted_labels) / data.shape[0]
    return score
