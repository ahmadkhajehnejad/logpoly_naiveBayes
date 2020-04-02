from logpoly.tools import  mp_compute_poly, mp_log_integral_exp, mp_compute_SS, mp_integral, mp_moments
import logpoly
import numpy as np
import sys

class Client:

    def __init__(self, data, labels, features_info):
        self.data = data
        self.labels = labels
        self.scaled_data = np.zeros(self.data.shape)
        for i in range(data.shape[1]):
            self.scaled_data[:,i] = logpoly.tools.scale_data(self.data[:,i], features_info[i]['min_value'], features_info[i]['max_value'])
        self.sent_bytes = 0
        self.communication_rounds = 0

    def get_n(self):
        self.communication_rounds += 1
        result = self.scaled_data.shape[0]
        self.sent_bytes += sys.getsizeof(result)

        print('        sending ', sys.getsizeof(result), ' bytes.')

        return result


    def get_logpoly_SS(self, d, c, k, from_, to_):
        self.communication_rounds += 1

        ax = np.arange(self.data.shape[0])
        ind = np.where( (self.labels == c) & (ax >= from_) & (ax < to_))[0]
        result = mp_compute_SS(self.scaled_data[ind,d], k)
        self.sent_bytes += sys.getsizeof(result)

        print('        sending ', sys.getsizeof(result), ' bytes.')

        return result

    def get_data(self, dimension, class_):
        self.communication_rounds += 1
        result = self.data[self.labels == class_, dimension]
        self.sent_bytes += sys.getsizeof(result)

        print('        sending ', sys.getsizeof(result), ' bytes.')

        return result
