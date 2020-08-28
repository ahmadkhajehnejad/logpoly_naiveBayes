import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from logpoly.model import  Logpoly
from logpoly.tools import mp_compute_SS
from logpoly.tools import scale_data
import config.logpoly

dt = pd.read_csv('../../data/HT_Sensor/HT_Sensor.csv').values
labels = dt[:,-1]
dt = dt[:,:-1]
dt_0 = dt[labels == 0, :]
dt_1 = dt[labels == 1, :]

dim = 9
plt.hist(dt_0[:, dim], bins=1000, color='blue', alpha=0.5, density=True)
plt.hist(dt_1[:, dim], bins=1000, color='red', alpha=0.5, density=True)

for i in range(10):
    print('#', i)
    print(np.min(dt_0[:,i]), np.min(dt_1[:,i]))
    print(np.max(dt_0[:, i]), np.max(dt_1[:, i]))
    print()

min_dt = np.min(dt[:, dim])
max_dt = np.max(dt[:, dim])
scaled_dt = scale_data(dt, min_dt, max_dt)
scaled_dt_0 = scaled_dt[labels == 0, :]
scaled_dt_1 = scaled_dt[labels == 1, :]

total_margin = config.logpoly.left_margin + config.logpoly.right_margin
# l = (max_dt - min_dt) / (1 - total_margin)
# x_points = np.arange(min_dt - l * config.logpoly.left_margin, max_dt + l * config.logpoly.right_margin, l/100 - 1e-7)
l = max_dt - min_dt
x_points = np.arange(min_dt, max_dt, l/100 - 1e-7)
x_points_scaled = scale_data(x_points, min_dt, max_dt)

k=10

logpoly_0 = Logpoly()
SS_0 = mp_compute_SS(scaled_dt_0[:, dim], k)
logpoly_0.fit(SS_0, scaled_dt.shape[0])
y_0 = np.exp(logpoly_0.logpdf(x_points_scaled)) / l
plt.plot(x_points, y_0, color='blue') #, alpha=0.5)

logpoly_1 = Logpoly()
SS_1 = mp_compute_SS(scaled_dt_1[:, dim], k)
logpoly_1.fit(SS_1, scaled_dt.shape[0])
y_1 = np.exp(logpoly_1.logpdf(x_points_scaled)) / l
plt.plot(x_points, y_1, color='red') #, alpha=0.5)

plt.show()
