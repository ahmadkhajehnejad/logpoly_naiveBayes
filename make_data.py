import config.general
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import matplotlib.pyplot as plt


if not(os.path.isdir('./data')):
    os.mkdir('./data')

if config.general.dataset_name == 'synthetic':
    n = 1000
    #_pi = [ 0.2, 0.3, 0.3, 0.2]
    #mu = [0,3,10,14]
    #var = [1,4,3,2]
    gmm = GaussianMixture(n_components=4, max_iter=2)
    gmm.fit(np.array([-0.5,0.5,2,3,4,9,10,11,13.5,14.5]).reshape([-1,1]))
    #gmm.set_params( weights=np.array(_pi), means=np.array(mu).reshape([-1,1]), precisions=np.array([1/a for a in var]).reshape([-1,1,1]))
    x, _ = gmm.sample(n)
    x = x.reshape([-1])

np.save('./data/' + config.general.dataset_name + '.npy', x)

plt.hist(x, bins=100)
plt.show()
