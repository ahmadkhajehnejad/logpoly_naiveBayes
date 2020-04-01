import numpy as np

import config.classifier


def get_train_and_validation_index(ind):
    n_total = ind.size
    if config.classifier.smart_validation:
        index_validation = ind[
            np.arange(config.classifier.validation_portion - 1, n_total, config.classifier.validation_portion)]
        index_tmp = np.ones([n_total])
        index_tmp[index_validation] = 0
        index_train = ind[np.where(index_tmp)[0]]
    else:
        np.random.shuffle(ind)
        n_validation = n_total // config.classifier.validation_portion
        index_validation = ind[:n_validation]
        index_train = ind[n_validation:]

    return [index_train, index_validation]