import numpy as np
import config.general, config.logpoly
import matplotlib.pyplot as plt

def show_data():
    x = np.load('./data/' + config.general.dataset_name + '.npy')
    min_x = np.min(x,axis=0)
    max_x = np.max(x,axis=0)
    x = ( (x - min_x) / (max_x - min_x) ) * (0.9 * (config.logpoly.x_ubound - config.logpoly.x_lbound)) + config.logpoly.x_lbound + (0.05 * (config.logpoly.x_ubound - config.logpoly.x_lbound))
    l = config.logpoly.x_ubound - config.logpoly.x_lbound
    plt.hist(x,np.arange(config.logpoly.x_lbound + (l/200), config.logpoly.x_ubound, l/100))    
    
show_data()