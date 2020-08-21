import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

a = tf.Variable([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
b = tf.constant([[0., np.nan, 0.], [np.nan, np.nan, 2.]])
y = tf.constant([0.,1.])
with tf.GradientTape() as gt:
    c = a - tf.expand_dims(b, -2)
    d = tf.where(tf.math.is_nan(c), tf.zeros_like(c), c)
    aw = tf.exp(-tf.reduce_sum(tf.math.square(d), -1))
    p = tf.reduce_sum(aw, -1)
    error = tf.keras.losses.mse(y,p)


print(gt.gradient(error, a))
