import numpy as np


class CategoricalDensityEstimator:

    def __init__(self, x, categories):
        self.theta = {}
        for c in categories:
            self.theta[c] = np.sum(np.array(x) == c)
        sum_counts = np.sum(self.theta.values())
        for c in categories:
            self.theta[c] = self.theta[c] / sum_counts

    def logpdf(self, x):
        return np.array([self.theta[c] for c in x])