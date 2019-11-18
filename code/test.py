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

# x = tf.placeholder(tf.float64, shape=[3])
# y = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
# s = tf.Session()
# s.run(tf.global_variables_initializer())
# print(s.run(y, feed_dict={x: np.array([np.nan, 2.0, np.nan])}))

ds = fetch_openml('hepatitis')
target = ds['target']
tn = np.array(list(set(target)))
target = np.expand_dims(target, -1)
print(np.array(target == tn, dtype=np.float))
