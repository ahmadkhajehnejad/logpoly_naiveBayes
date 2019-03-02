
import logpoly
from logpoly.model import LogpolyModelSelector
from categorical.model import CategoricalDensityEstimator
from kernel_density_estimation.model import KDE, select_KDE_model
from Gaussian_mixture_model.model import select_GMM_model
import config.classifier
import config.general
import config.kde
import config.gmm
import numpy as np
from multiprocessing import Process, Queue
import gc
import sys


def fit_thread_func(shared_space, data, feature_info, c_index, i):
    print(c_index, i)
    sys.stdout.flush()
    scaled_data = logpoly.tools.scale_data(data, feature_info['min_value'],
                                           feature_info['max_value'])
    if feature_info['feature_type'] == config.general.CONTINUOUS_FEATURE:
        if config.classifier.continuous_density_estimator == 'logpoly':
            logpoly_model_selector = LogpolyModelSelector(config.logpoly.list_factor_degrees)
            shared_space.put([c_index, i, logpoly_model_selector.select_model(scaled_data)])
        elif config.classifier.continuous_density_estimator == 'kde':
            shared_space.put([c_index, i, KDE(scaled_data, bandwidth=None)])
        elif config.classifier.continuous_density_estimator == 'vkde':
            shared_space.put([c_index, i, select_KDE_model(scaled_data, config.kde.list_of_bandwidths)])
        elif config.classifier.continuous_density_estimator == 'gmm':
            shared_space.put([c_index, i, select_GMM_model(scaled_data, config.gmm.list_of_num_components)])

    elif feature_info['feature_type'] == config.general.CATEGORICAL_FEATURE:
        shared_space.put([c_index, i, CategoricalDensityEstimator(data, feature_info['categories'])])
    else:
        raise 'not handled case feature_type=' + str(feature_info['feature_type']) + \
              ' (dimension #' + str(i) + ')'


class NaiveBayesClassifier:

    def __init__(self, features_info):
        self.features_info = features_info
        self.density_estimators = None
        self.classes = self.features_info[-1]['classes']

    def fit(self, data, labels):
        self.density_estimators = [[None for _ in range(len(self.features_info)-1)] for _ in self.classes]

        if config.classifier.multiprocessing:
            shared_space = Queue()
            processes = []

            for c_index, c in enumerate(self.classes):
                class_index = labels == c
                class_data = data[class_index, :]
                for i in range(len(self.features_info) - 1):
                    processes.append(Process(target=fit_thread_func,
                                             args=[shared_space, class_data[:, i], self.features_info[i],
                                              c_index, i], daemon=True))

            head = np.min([config.general.max_num_processes, len(processes)])
            for i in range(head):
                processes[i].start()

            num_terminated = 0
            while num_terminated < len(processes):
                res = shared_space.get()
                c_index, i = res[0], res[1]
                self.density_estimators[c_index][i] = res[2]
                p_index = (len(self.features_info) - 1) * c_index + i
                processes[p_index].join()
                processes[p_index].terminate()
                processes[p_index] = None
                num_terminated += 1
                if np.mod(num_terminated, config.general.num_processes_per_garbage_collection) == 0:
                    gc.collect()
                if head < len(processes):
                    processes[head].start()
                    head += 1

            shared_space.close()
        else:
            for c_index, c in enumerate(self.classes):
                class_index = labels == c
                class_data = data[class_index, :]
                for i in range(len(self.features_info) - 1):
                    # print(c_index, i)
                    # sys.stdout.flush()
                    if self.features_info[i]['feature_type'] == config.general.CONTINUOUS_FEATURE:
                        scaled_data = logpoly.tools.scale_data(class_data[:, i], self.features_info[i]['min_value'],
                                                               self.features_info[i]['max_value'])

                        if config.classifier.continuous_density_estimator == 'logpoly':
                            logpoly_model_selector = LogpolyModelSelector(config.logpoly.list_factor_degrees)
                            self.density_estimators[c_index][i] = logpoly_model_selector.select_model(scaled_data)
                        elif config.classifier.continuous_density_estimator == 'kde':
                            self.density_estimators[c_index][i] = KDE(scaled_data, bandwidth=None)
                        elif config.classifier.continuous_density_estimator == 'vkde':
                            self.density_estimators[c_index][i] = select_KDE_model(scaled_data, config.kde.list_of_bandwidths)
                        elif config.classifier.continuous_density_estimator == 'gmm':
                            self.density_estimators[c_index][i] = select_GMM_model(scaled_data, config.gmm.list_of_num_components)

                    elif self.features_info[i]['feature_type'] == config.general.CATEGORICAL_FEATURE:
                        self.density_estimators[c_index][i] = CategoricalDensityEstimator(class_data[:, i],
                                                                                          self.features_info[i][
                                                                                              'categories'])
                    else:
                        raise 'not handled case feature_type=' + str(self.features_info[i]['feature_type']) + \
                              ' (dimension #' + str(i) + ')'

        print('fit')
        sys.stdout.flush()


    def get_log_likelihood_per_class(self, data):
        log_likelihood_per_class = np.zeros([data.shape[0], len(self.classes)])
        for c_index, c in enumerate(self.classes):
            log_likelihood_per_dim = np.zeros([data.shape[0], len(self.features_info) - 1])
            for i in range(len(self.features_info) - 1):
                if self.features_info[i]['feature_type'] == config.general.CONTINUOUS_FEATURE:
                    scaled_data = logpoly.tools.scale_data(data[:, i], self.features_info[i]['min_value'],
                                                           self.features_info[i]['max_value'])
                    log_likelihood_per_dim[:, i] = self.density_estimators[c_index][i].logpdf(scaled_data)
                else:
                    log_likelihood_per_dim[:, i] = self.density_estimators[c_index][i].logpdf(data[:, i])
            log_likelihood_per_class[:, c_index] = np.sum(log_likelihood_per_dim, axis=1)
        return log_likelihood_per_class

    def label(self, data):
        log_likelihood_per_class = self.get_log_likelihood_per_class(data)
        predicted_labels = np.argmax(log_likelihood_per_class, axis=1)
        return self.classes[predicted_labels]

    def get_params(self, deep=False):
        return {'features_info': self.features_info}

def scorer_thread_func( shared_space, l, r, classifier, data):
    print(l,r)
    shared_space.put( [l, r, classifier.label(data)] )


def scorer(classifier, data, labels):
    if config.classifier.multiprocessing:

        n_test = data.shape[0]
        shared_space = Queue()
        part_size = config.classifier.max_test_size
        processes = [None] * ( (n_test + part_size - 1) // part_size )

        for i in range(len(processes)):

            l_index = i * part_size
            if i == len(processes) - 1:
                r_inddex = n_test
            else:
                r_inddex = (i+1) * part_size
            processes[i] = Process(target=scorer_thread_func, args=[shared_space, l_index, r_inddex, classifier,
                                                                    data[l_index:r_inddex, :].reshape(
                                                                        [-1, data.shape[1]])], daemon=True)

        print('num of all scoring processes:', len(processes))
        print('size of test data:', n_test)
        sys.stdout.flush()

        predicted_labels = np.zeros([n_test], dtype=classifier.classes.dtype)

        head = np.min([config.general.max_num_processes, len(processes)])
        for i in range(head):
            processes[i].start()

        num_terminated = 0
        while num_terminated < len(processes):
            res = shared_space.get()
            l_index, r_index = res[0], res[1]
            predicted_labels[l_index:r_index] = res[2]
            p_index = l_index // part_size
            processes[p_index].join()
            processes[p_index].terminate()
            processes[p_index] = None
            num_terminated += 1
            if np.mod(num_terminated, config.general.num_processes_per_garbage_collection) == 0:
                gc.collect()
            if head < len(processes):
                processes[head].start()
                head += 1

        shared_space.close()
    else:
        predicted_labels = classifier.label(data)
    score = np.sum(labels == predicted_labels) / data.shape[0]
    return score
