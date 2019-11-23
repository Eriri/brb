import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml

mae = []
for i in range(10):
    res = np.load('eval%d.npy' % i)
    for bres in res:
        mae.append(bres[:, 0])
mae = np.array(mae)
print(np.max(mae, 0))
print(np.mean(mae, 0))
print(np.min(mae, 0))
epoch = np.arange(0, 16, 1)
