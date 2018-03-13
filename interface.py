import numpy as np
import config
from tools import compute_SS

class Interface:
    def __init__(self):
        x = np.load('../data/' + config.dataset_name + '/data/npy')
        self.n = len(x)
        min_x = np.min(x,axis=0)
        max_x = np.max(x,axis=0)
        self.x = ( (x - min_x) / (max_x - min_x) ) * (0.9 * (config.x_ubound - config.x_lbound)) + config.x_lbound + (0.05 * (config.x_ubound - config.x_lbound))
        
    def get_SS(self, theta):
        return compute_SS(self.x,theta)
    
    def get_n(self):
        return self.n