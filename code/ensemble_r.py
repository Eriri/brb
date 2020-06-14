import os
import tensorflow as tf
import numpy as np
from dataset import dataset_oil
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from util import kfold, generate_variable, generate_constant

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Model(tf.keras.Model):
    def __init__(self, rule_num, att_dim, res_dim, out_util):
        super(Model, self).__init__()
        self.a = generate_variable((rule_num, att_dim,))
        self.d = generate_variable((att_dim,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))
        self.u = tf.constant(out_util)
        self.compile(tf.keras.optimizers.Adam(), tf.keras.losses.MSE, ['mae'])

    def call(self, inputs):
        w = tf.math.square(self.a - tf.expand_dims(inputs, -2)) * tf.math.exp(self.d)
        aw = tf.exp(-tf.reduce_sum(tf.where(tf.math.is_nan(w), tf.zeros_like(w), w), -1)) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        out = tf.reduce_sum(pc * self.u, -1)
        return tf.where(tf.math.is_nan(out), tf.reduce_mean(self.u), out)


def main():


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    main()
