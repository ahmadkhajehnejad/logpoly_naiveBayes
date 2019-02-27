from sklearn.mixture.gaussian_mixture import GaussianMixture as GMM
import numpy as np
import config.gmm


class GaussianMixtureModel:

    def __init__(self, data, num_components):
        self.gmm = GMM(n_components=num_components, covariance_type='full', max_iter=config.gmm.max_iter, n_init=config.gmm.n_init)
        self.gmm.fit(np.array(data).reshape([-1,1]))


    def logpdf(self, x):
        return np.array(self.gmm.score_samples(np.array(x).reshape([-1, 1])))


def select_GMM_model(data, list_of_num_components):

    n_total = data.shape[0]

    ind = np.argsort(data)
    index_validation = ind[np.arange(3, n_total, 4)]
    index_tmp = np.ones([n_total])
    index_tmp[index_validation] = 0
    index_train = ind[np.where(index_tmp)[0]]

    # ind = np.arange(n_total)
    # np.random.shuffle(ind)
    # n_train = n_total // 4
    # index_train = ind[:n_train]
    # index_validation = ind[n_train:]

    gmm_models = []
    avg_log_likelihoods = []
    for i, k in enumerate(list_of_num_components):
        gmm_models.append(GaussianMixtureModel(data[index_train], num_components=k))
        avg_log_likelihoods.append(np.mean(gmm_models[i].logpdf(data[index_validation])))

    best_index = np.argmax(avg_log_likelihoods)
    return GaussianMixtureModel(data, num_components=list_of_num_components[best_index])
