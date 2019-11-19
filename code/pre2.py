import tensorflow as tf
import numpy as np
from util import *


class BRB:
    def __init__(self, rule_num, att_dim, res_dim, low, high, one, util):
        self.A = tf.Variable(np.random.uniform(low, high, size=(rule_num, att_dim,)), dtype=tf.float64, trainable=True)
        self.D = tf.Variable(np.log(one), dtype=tf.float64, trainable=True)
        self.B = tf.Variable(tf.random.normal(shape=(rule_num, res_dim,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.R = tf.Variable(tf.zeros(shape=(rule_num,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.U = tf.Variable(util, dtype=tf.float64, trainable=True)

    @tf.function
    def rimer(self, x):
        w = tf.math.square((self.A - tf.expand_dims(x, -2)) / tf.math.exp(self.D))
        aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.R)
        sw = tf.reduce_sum(aw, -1)
        bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(self.B)+1.0, -2)-1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        out = tf.reduce_sum(pc*self.U, -1)
        return out

    def train(self, x, y, ep, bs):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.shuffle(1000).batch(bs).repeat(ep)
        opt = tf.keras.optimizers.Adam()
        for tx, ty in ds:
            err = tf.keras.losses.MSE(ty, self.rimer(tx))
            opt.minimize(lambda: err, [self.A, self.D, self.B, self.R, self.U])
            print(err)


def main():
    x, y = read_oil()
    low, high, one = np.min(x, 0), np.max(x, 0), np.ptp(x, 0)
    util = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    brb = BRB(64, 2, 5, low, high, one, util)
    brb.train(x, y, 100, 64)


if __name__ == "__main__":
    main()
