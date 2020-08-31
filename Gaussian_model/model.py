import numpy as np
import scipy


class GaussianModel:

    def __init__(self, data):
        self.mu = np.mean(data)
        if np.unique(data).shape[0] > 1:
            self.std = np.std(data)
        else:
            self.std = 1e-8

        if np.isnan(self.mu):
            print(data)

    def logpdf(self, x):
        return scipy.stats.norm.logpdf(x, loc=self.mu, scale=self.std)

