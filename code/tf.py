import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import tqdm
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
les = np.arange(8)


def trans(e, l):
    z = np.zeros(len(l))
    for i in range(len(l)):
        z[i] = float(l[i] == e)
    for i in range(len(l)-1):
        if l[i] < e and e < l[i+1]:
            z[i], z[i+1] = l[i+1] - e, e-l[i]
    return z


def ReadData(filename):
    ant, con = [], []
    with open(file=filename, mode='r') as f:
        for i in f:
            e = list(map(float, i.strip().split()))
            ant.append(e[:-1]), con.append(trans(e[-1], les))
    one, low, high = np.ptp(ant, axis=0), np.min(ant, axis=0), np.max(ant, axis=0)
    return np.array(ant), np.array(con), one, low, high


class EBRB:
    def __init__(self, one, low, high, les, att_dim, res_dim, rule_num):
        self.TA, self.TC = tf.compat.v1.placeholder(tf.float64, [None, att_dim]), tf.compat.v1.placeholder(tf.float64, [None, res_dim])
        # self.ONE = tf.constant(one, dtype=tf.float64)
        # self.OT = tf.Variable(tf.random.normal(one.shape, dtype=tf.float64), trainable=True)
        # self.O = self.ONE * tf.math.log_sigmoid(self.ONE)
        self.O = tf.Variable(tf.math.log(one), expected_shape=[att_dim], trainable=True)
        # self.O = tf.Variable(one, expected_shape=[att_dim], trainable=True)
        self.A = tf.Variable(np.random.uniform(low, high, (rule_num, att_dim)), dtype=tf.float64, trainable=True)
        self.C = tf.Variable(tf.random.normal((rule_num, res_dim,), dtype=tf.float64), trainable=True)
        self.RW = tf.Variable(tf.random.normal((rule_num,), dtype=tf.float64), trainable=True)
        self.L = tf.constant(les, dtype=tf.float64, shape=(res_dim,), verify_shape=True)
        self.W = tf.exp(-tf.reduce_sum(tf.square((self.A-self.TA[:, None])/tf.math.exp(self.O)), axis=2))*tf.math.exp(self.RW)+tf.cast(1e-10, tf.float64)
        self.SW = tf.reduce_sum(self.W, 1)
        self.BC = tf.nn.softmax(self.C, -1)
        self.B = tf.reduce_prod(tf.expand_dims(self.W/(self.SW[:, None]-self.W), -1)*self.BC+tf.cast(1, tf.float64), -2)-1
        self.PC = self.B / tf.reduce_sum(self.B, -1)[:, None]
        self.ERROR = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.TC, self.PC))
        # self.ERROR = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.TC, self.PC))
        self.TRUE, self.PRED = tf.reduce_sum(self.TC * self.L, -1), tf.reduce_sum(self.PC * self.L, -1)
        self.MAE = tf.reduce_mean(tf.abs(self.TRUE - self.PRED))
        self.MSE = tf.reduce_mean(tf.square(self.TRUE - self.PRED))
        self.STEP = tf.compat.v1.train.AdamOptimizer().minimize(self.ERROR)
        self.SESS = tf.compat.v1.Session()
        self.SESS.run(tf.compat.v1.global_variables_initializer())

    def train(self, ant, con, ta, tc, ep=10000, bs=64):
        cb = int(np.floor(len(ant)/bs))
        for e in range(ep):
            ac = list(zip(ant, con))
            random.shuffle(ac)
            ant, con = zip(*ac)
            for b in range(cb):
                self.SESS.run(self.STEP, {self.TA: ant[b*bs:(b+1)*bs], self.TC: con[b*bs:(b+1)*bs]})
            mae = self.SESS.run(self.MAE, {self.TA: ant, self.TC: con})
            tmae = self.SESS.run(self.MAE, {self.TA: ta, self.TC: tc})
            print(e, mae, tmae)

    def predict(self, ant, con):
        mae, mse = self.SESS.run([self.MAE, self.MSE], {self.TA: ant, self.TC: con})
        print(mae, mse)
        return (mae, mse)


def main():
    ant, con, one, low, high = ReadData('../data/oil.data')
    skf = KFold(n_splits=10, shuffle=True, random_state=0)
    res = []
    for train_mask, test_mask in skf.split(ant):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]
        ebrb = EBRB(one, low, high, les, ant.shape[1], con.shape[1], 16)
        ebrb.train(train_ant, train_con, test_ant, test_con)
        res.append(ebrb.predict(test_ant, test_con))
    print(res)


if __name__ == "__main__":
    main()
