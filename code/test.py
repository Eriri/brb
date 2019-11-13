import tensorflow as tf
import numpy as np

a, b = tf.Variable(np.array([[0.0, 1.0], [1.0, 0.0]])), tf.Variable(np.array([0.5, 0.5]))
x, y = tf.distributions.Categorical(logits=a), tf.distributions.Categorical(logits=b)
z = tf.distributions.kl_divergence(y, x)
d = tf.keras.losses.categorical_crossentropy(b, a)
s = tf.Session()
s.run(tf.global_variables_initializer())
print(s.run(d))
