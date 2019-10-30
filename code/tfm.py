import tensorflow as tf
import numpy as np
import random
import tqdm
import math
import os
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class EBRB():
    def __init__(self, one, low, high, adim, rdim, rule_num, X):  # (None, adim)
        self.A = tf.Variable(np.random.uniform(low, high, (rule_num, adim,)),
                             expected_shape=[rule_num, adim], dtype=tf.float64, trainable=True)
        self.C = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, rdim,)),
                             expected_shape=[rule_num, rdim], dtype=tf.float64, trainable=True)
        self.D = tf.Variable(np.log(one),
                             expected_shape=[adim], dtype=tf.float64, trainable=True)
        self.W = tf.Variable(np.random.normal(0.0, 1.0, (rule_num,)),
                             expected_shape=[rule_num], dtype=tf.float64, trainable=True)

        self.BC, self.RW, self.O = tf.nn.softmax(self.C), tf.exp(self.W), tf.math.exp(self.D)

        self.AW = tf.exp(-tf.reduce_sum(tf.math.square((self.A-tf.expand_dims(X, -2))/self.O), -1)) * self.RW
        # (rule_num,adim) - (None,1,adim) = (None,rule_num,adim)
        # reduce_sum((None,rule_num,adim),-1) = (None,rule_num)

        self.SW = tf.reduce_sum(self.AW, -1)
        # reduce_sum((None,rule_num),-1) = (None)

        self.B = tf.reduce_prod(tf.expand_dims(self.AW/(tf.expand_dims(self.SW, -1)-self.AW), -1)*self.BC+1.0, -2)-1.0
        # (None,1) - (None,rule_num) = (None,rule_num)
        # reduce_prod((None,rule_num,rdim),-2) = (None,rdim)

        self.Y = self.B/tf.expand_dims(tf.reduce_sum(self.B, -1), -1)
        # (None,rdim) / (None,1) = (None,rdim)


class DBRB():
    def __init__(self, anum, adim, rdim, rule_num, X):  # (None, anum, adim)
        self.A = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, anum, adim,)),
                             expected_shape=[rule_num, anum, adim], dtype=tf.float64, trainable=True)
        self.C = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, rdim,)),
                             expected_shape=[rule_num, rdim], dtype=tf.float64, trainable=True)
        self.W = tf.Variable(np.random.normal(0.0, 1.0, (rule_num,)),
                             expected_shape=[rule_num], dtype=tf.float64, trainable=True)

        self.DA, self.BC, self.RW = tf.nn.softmax(self.A), tf.nn.softmax(self.C), tf.exp(self.W)

        self.AW = tf.reduce_max(tf.reduce_sum(tf.sqrt(self.DA * tf.expand_dims(X, -3)), -1), -1) * self.RW
        # (rule_num,anum,adim) * (None,1,anum,adim) = (None,rule_num,anum,adim)
        # reduce_sum((None,rule_num,anum,adim),-1) = (None,rule_num,anum)
        # reduce_max((None,rule_num,anum),-1) = (None,rule_num)

        self.SW = tf.reduce_sum(self.AW, -1)
        # reduce_sum((None,rule_num),-1) = (None)

        self.B = tf.reduce_prod(tf.expand_dims(self.AW/(tf.expand_dims(self.SW, -1)-self.AW), -1)*self.BC+1.0, -2)-1.0
        # (None,rule_num,1) * (rule_num,rdim) = (None,rule_num,rdim)
        # reduce_prod((None,rule_num,rdim),-2) = (None,rdim)

        self.Y = self.B/tf.expand_dims(tf.reduce_sum(self.B, -1), -1)
        # (None,rdim) / (None,1) = (None,rdim)


class BRB():
    def __init__(self, one, low, high, les, dim_1, dim_2, dim_3, e_num, e_rn, d_rn):
        self.X = tf.compat.v1.placeholder(tf.float64, [None, dim_1])
        self.Y = tf.compat.v1.placeholder(tf.float64, [None, dim_3])

        self.E = [EBRB(one, low, high, dim_1, dim_2, e_rn, self.X) for i in range(e_num)]
        self.EO = tf.compat.v2.concat([tf.expand_dims(e.Y, -2) for e in self.E], -2)
        self.D = DBRB(e_num, dim_2, dim_3, d_rn, self.EO)

        self.ERROR = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.Y, self.D.Y))
        self.STEP = tf.compat.v1.train.AdamOptimizer().minimize(self.ERROR)
        self.ACC = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.Y, -1), tf.argmax(self.D.Y, -1)), tf.float64))

        self.L = tf.constant(les, dtype=tf.float64, shape=(dim_3,), verify_shape=True)
        self.TRUE, self.PRED = tf.reduce_sum(self.Y * self.L, -1), tf.reduce_sum(self.D.Y * self.L, -1)
        self.MAE = tf.reduce_mean(tf.abs(self.TRUE - self.PRED))
        self.MSE = tf.reduce_mean(tf.square(self.TRUE - self.PRED))

        self.SESS = tf.compat.v1.InteractiveSession()
        self.SESS.run(tf.compat.v1.global_variables_initializer())

    def train(self, oa, oc, ta, tc, ep=10000, bs=16):
        cb = int(len(oa)/bs)

        for e in range(ep):
            pb = tqdm.tqdm(total=cb)
            oac = list(zip(oa, oc))
            random.shuffle(oac)
            oa, oc = zip(*oac)
            for b in range(cb):
                self.SESS.run(self.STEP, {self.X: oa[b*bs:(b+1)*bs], self.Y: oc[b*bs:(b+1)*bs]})
                pb.update()
            pb.close()
            print(e, self.SESS.run([self.ERROR, self.MAE, self.MSE], {self.X: oa, self.Y: oc}))
            print(self.SESS.run([self.MAE, self.MSE], {self.X: ta, self.Y: tc}))

    def predict(self, a, c):
        return self.SESS.run(self.ACC, {self.X: a, self.Y: c})


class Model():
    def __init__(self, one, low, high, adim, rdim, rule_num):
        self.X = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, adim])
        self.Y = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None, rdim])
        self.W = tf.compat.v1.placeholder(dtype=tf.float64, shape=[None])
        self.E = EBRB(one, low, high, adim, rdim, rule_num, self.X)
        self.ERROR = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.Y, self.E.Y)*self.W)
        self.STEP = tf.train.AdamOptimizer().minimize(self.ERROR)
        self.PRED = tf.argmax(self.E.Y)
        self.ACC = tf.reduce_mean(tf.cast(tf.equal(self.PRED, tf.argmax(self.E.Y)), tf.float64))
        self.SESS = tf.compat.v1.InteractiveSession()
        self.SESS.run(tf.compat.v1.global_variables_initializer())

    def train(self, ant, con, wei, ep=5, bs=64):
        bn = int(math.ceil(len(ant)/bs))
        for i in range(bn):

    def predict(self, ant):
        return self.SESS.run(self.PRED)


def main():
    pass


if __name__ == "__main__":
    main()
