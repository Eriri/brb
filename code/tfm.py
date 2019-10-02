import tensorflow as tf
import numpy as np


class EBRB():
    def __init__(self, one, low, high, adim, rdim, rule_num, X):
        self.A = tf.Variable(np.random.uniform(low, high, (rule_num, adim,)), dtype=tf.float64, trainable=True)
        self.C = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, rdim,)), dtype=tf.float64, trainable=True)
        self.O = tf.Variable(np.log(one), dtype=tf.float64, trainable=True)
        self.W = tf.Variable(np.random.normal(0.0, 1.0, (rule_num,)), dtype=tf.float64, trainable=True)

        self.BC, self.RW = tf.nn.softmax(self.C), tf.exp(self.W)

        self.AW = tf.exp(-tf.reduce_sum(tf.math.square((self.A-tf.expand_dims(X, -2))/self.O), -1)) * self.RW
        # (rule_num,adim) - (None,1,adim) = (None,rule_num,adim), reduce_sum((None,rule_num,adim),-1) = (None,rule_num)

        self.SW = tf.reduce_sum(self.AW, -1)
        # reduce_sum((None,rule_num),-1) = (None)

        self.B = tf.reduce_prod(tf.expand_dims(self.AW/(tf.expand_dims(self.SW, -1)-self.AW), -1)*self.BC+1.0, -2)-1.0
        # (None,rule_num,1) * (rule_num,rdim) = (None,rule_num,rdim), reduce_prod((None,rule_num,rdim),-2) = (None,rdim)

        self.Y = self.B/tf.expand_dims(tf.reduce_sum(self.B, -1), -1)
        # (None,rdim) / (None,1) = (None,rdim)


class DBRB():
    def __init__(self, anum, adim, rdim, rule_num, X):
        self.A = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, anum, adim,)), dtype=tf.float64, trainable=True)
        self.C = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, rdim,)), dtype=tf.float64, trainable=True)
        self.W = tf.Variable(np.random.normal(0.0, 1.0, (rule_num,)), dtype=tf.float64, trainable=True)

        self.DA, self.BC, self.RW = tf.nn.softmax(self.A), tf.nn.softmax(self.C), tf.exp(self.W)


class BRB():
    def __init__(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
