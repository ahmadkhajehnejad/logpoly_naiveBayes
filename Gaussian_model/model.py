import numpy as np
import scipy


class GaussianModel:

    def __init__(self, data):
        self.mu = np.mean(data)
        self.std = np.std(data)

        if np.isnan(self.mu):
            print(data)

    def logpdf(self, x):
        return scipy.stats.norm.logpdf(x, loc=self.mu, scale=self.std)

