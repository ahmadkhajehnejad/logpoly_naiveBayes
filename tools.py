import numpy as np

import config.classifier


def get_train_and_validation_index(ind):
    n_total = ind.size
    if config.classifier.smart_validation:
        index_validation = ind[
            np.arange(config.classification.validation_portion - 1, n_total, config.classification.validation_portion)]
        index_tmp = np.ones([n_total])
        index_tmp[index_validation] = 0
        index_train = ind[np.where(index_tmp)[0]]
    else:
        np.random.shuffle(ind)
        n_train = n_total // config.classification.validation_portion
        index_train = ind[:n_train]
        index_validation = ind[n_train:]

    return [index_train, index_validation]