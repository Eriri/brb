import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml


ds = fetch_openml('adult', 2)
print(ds.keys())
print(ds['feature_names'])
print(ds['data'])
