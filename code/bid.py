import os
import tensorflow as tf
import numpy as np
from dataset import dataset_oil
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from util import kfold, generate_variable, generate_constant

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class BModel(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim):
        super(BModel, self).__init__()
        self.a = generate_variable((rule_num, att_dim,))
        self.d = generate_variable((att_dim,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))

    def __call__(self, x):
        w = tf.math.exp(-tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d))
        aw = tf.reduce_sum(tf.where(tf.math.is_nan(w), tf.zeros_like(w), w), -1) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.where(tf.math.is_nan(pc), tf.zeros_like(pc), pc)


class AModel(tf.Module):
    def __init__(self, rule_num, base_num, base_rule_num, att_dim, mid_dim, res_dim):
        super(AModel, self).__init__()
        self.bms = [BModel(base_rule_num, att_dim, mid_dim) for _ in range(base_num)]
        self.a = generate_variable((rule_num, base_num, mid_dim,))
        self.d = generate_variable((base_num,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))

    def __call__(self, x):
        mid_out = tf.concat([tf.expand_dims(bm(x), -2) for bm in bms], -2)
        w = tf.keras.losses.categorical_crossentropy(tf.expand_dims(mid_out, -3), self.a) * tf.math.exp(self.d)
        aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.where(tf.math.is_nan(pc), tf.zeros_like(pc), pc)


def main():
    pass


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    main()
