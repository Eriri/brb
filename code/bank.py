import tensorflow as tf
import numpy as np
import pandas as pd


class EBRB:  # metric_shape category_shapes one low high rule_num result_shape
    def __init__(self, ms, cs, on, lo, hi, rn, rs):
        self.xm = tf.placeholder(dtype=tf.float64, shape=[None, ms])
        self.xc = [tf.placeholder(dtype=tf.float64, shape=[None, s]) for s in cs]
        self.yi = tf.placeholder(dtype=tf.float64, shape=[None, rs])

        self.am = tf.Variable(np.random.uniform(lo, hi, (rn, ms)), dtype=tf.float64)
        self.ac = [tf.Variable(np.random.normal(0.0, 1.0, (rn, s,)), dtype=tf.float64) for s in cs]
        self.on = tf.Variable(np.log(on), dtype=tf.float64)
        self.co = tf.Variable(np.random.normal(0.0, 1.0, (rn, rs)), dtype=tf.float64)
        self.rw = tf.Variable(np.random.normal(0.0, 1.0, (rn,)), dtype=tf.float64)

        dac, don, dco, drw = [tf.nn.softmax(a) for a in self.ac], tf.exp(self.on), tf.nn.softmax(self.co), tf.exp(self.rw)

        self.awm = tf.exp(-tf.reduce_sum(tf.square((self.am - tf.expand_dims(self.xm, -2))/don), -1)) * drw
        self.awc = tf.reduce_prod(tf.concat([tf.reduce_sum(tf.sqrt(a * tf.expand_dims(x, -2)), -1) for a, x in zip(dac, self.xc)], -1), -1)

        self.aw, self.sw = self.awm * self.awc, tf.reduce_sum(self.awm * self.awc, -1)
        self.bc = tf.reduce_prod(tf.expand_dims(self.aw/(tf.expand_dims(self.sw, -1)-self.aw), -1)*dco+1.0, -2)-1.0
        self.yo = self.bc/tf.expand_dims(tf.reduce_sum(self.bc, -1), -1)

        self.err = tf.keras.losses.categorical_crossentropy(self.yi, self.yo)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.yi, -1), tf.argmax(self.yo, -1)), tf.float64))

        self.step = tf.train.AdamOptimizer().minimize(self.err)
        self.sess = tf.Session()

    def train(self, ep=10000, bs=64):
        pass


def main():
    df = pd.read_csv('train_rev.csv')
    for


if __name__ == "__main__":
    main()
