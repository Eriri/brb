import tensorflow as tf
import numpy as np


class BRB:
    def __init__(self, one, att_dim, res_dim, base_it, base_num, base_ant, base_con):
        self.O = tf.Variable(one, expected_shape=[att_dim])
        self.A, self.C, self.W = [], [], []
        self.TA, self.TC, self.PC = [], [], []
        self.ERROR, self.ACC, self.STEP = [], [], [], []
        for i in range(base_it):
            self.A.append(tf.constant(base_ant[i*base_num:(i+1)*base_num]))
            self.C.append(tf.constant(base_con[i*base_num:(i+1)*base_num]))
            self.W.append(tf.Variable(tf.random_uniform((base_num), minval=-1, maxval=1, dtype=tf.float64)))
            self.TA.append(tf.placeholder(tf.float64, [None, att_dim]))
            self.TC.append(tf.placeholder(tf.float64, [None, res_dim]))
        for i in range(base_it):
            w = tf.exp(-tf.reduce_sum(tf.square((self.A[i]-self.TA[i])/self.O), axis=1))*tf.log_sigmoid(self.W[i])+1e-8
            sw = tf.reduce_sum(w)
            B = tf.reduce_prod((w/(sw-w))[:, tf.newaxis]*self.C[i]+1, axis=0)-1
            self.PC.append(B / tf.reduce_sum(B))
        for i in range(base_it):
            self.ERROR.append(tf.keras.losses.categorical_crossentropy(self.TC[i], self.PC[i]))
            self.PRED = tf.equal(tf.argmax(self.TC[i], axis=1), tf.argmax(self.PC[i], axis=1))
            self.ACC.append(tf.reduce_mean(tf.cast(self.PRED, tf.float64)))
        for i in range(base_it):
            self.STEP.append(tf.train.AdamOptimizer().minimize(self.ERROR[i]))
        self.S = tf.InteractiveSession()
        self.S.run(tf.global_variables_initializer())
        self.S.close()

    def train():
        pass


def main():


if __name__ == "__main__":
    main()
