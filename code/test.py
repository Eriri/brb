import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml


'''
a, b = tf.Variable(np.array([[-1.0, 1.0], [1.0, -1.0]])), tf.Variable(np.array([-2.0, 2.0]))
x, y = tf.distributions.Categorical(logits=a), tf.distributions.Categorical(logits=b)
z = tf.distributions.kl_divergence(y, x)
d = tf.keras.losses.categorical_crossentropy(b, a)
s = tf.Session()
s.run(tf.global_variables_initializer())
print(s.run(x))
'''

ds = fetch_openml('hepatitis')
print(ds['DESCR'])
