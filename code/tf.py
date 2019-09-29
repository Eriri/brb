import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score
from ADASYN import ADASYN
import random
import tqdm


def ReadData(filename):
    ant, con = [], []
    with open(file=filename, mode='r') as f:
        for i in f:
            e = list(map(float, i.split()))
            ant.append(e[:-1])
            t = [0.0, 0.0]
            t[int(int(e[-1]) != 1)] = 1.0
            con.append(t)
    one, low, high = np.ptp(ant, axis=0), np.min(ant, axis=0), np.max(ant, axis=0)
    return np.array(ant), np.array(con), one, low, high


class BRB:
    def __init__(self, one, low, high, att_dim, res_dim, rule_num):
        self.TA, self.TC = tf.placeholder(tf.float64, [None, att_dim]), tf.placeholder(tf.float64, [None, res_dim])
        self.O = tf.Variable(one, expected_shape=[att_dim], trainable=True)
        self.A = tf.Variable(np.random.uniform(low, high, (rule_num, att_dim)),
                             dtype=tf.float64, expected_shape=[rule_num, att_dim], trainable=True)
        self.C = tf.Variable(tf.random_normal((rule_num, res_dim,), dtype=tf.float64),
                             dtype=tf.float64, expected_shape=[rule_num, res_dim], trainable=True)
        self.RW = tf.Variable(tf.random_normal((rule_num,), dtype=tf.float64),
                              dtype=tf.float64, expected_shape=[rule_num], trainable=True)
        self.W = tf.exp(-tf.reduce_sum(tf.square((self.A-self.TA[:, None])/self.O), axis=2))*tf.log_sigmoid(self.RW)+tf.cast(1e-8, tf.float64)
        self.SW = tf.reduce_sum(self.W, axis=1)
        self.BC = tf.log_sigmoid(self.C) / tf.reduce_sum(tf.log_sigmoid(self.C), axis=-1)[:, None]
        self.B = tf.reduce_prod(tf.expand_dims(self.W/(self.SW[:, None]-self.W), -1)*self.BC+tf.cast(1, tf.float64), -2)-1
        self.PC = self.B / tf.reduce_sum(self.B, -1)[:, None]
        self.ERROR = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.TC, self.PC))
        self.PRED = tf.equal(tf.argmax(self.TC, -1), tf.argmax(self.PC, -1))
        self.ACC = tf.reduce_mean(tf.cast(self.PRED, tf.float64))
        self.STEP = tf.train.AdamOptimizer().minimize(self.ERROR)
        self.SESS = tf.Session()
        self.SESS.run(tf.global_variables_initializer())

    def train(self, ant, con, ep=10, bs=32):
        cb = int(np.floor(len(ant)/bs))
        pb = tqdm.tqdm(total=ep)
        for e in range(ep):
            ac = list(zip(ant, con))
            random.shuffle(ac)
            ant, con = zip(*ac)
            for b in range(cb):
                self.SESS.run(self.STEP, {self.TA: ant[b*bs:(b+1)*bs], self.TC: con[b*bs:(b+1)*bs]})
            pb.update()
        pc = self.SESS.run(self.PC, {self.TA: ant})
        print(e, accuracy_score(np.argmax(con, -1), np.argmax(pc, -1)), f1_score(np.argmax(con, -1), np.argmax(pc, -1)))
        pb.close()

    def predict(self, ant):
        return self.SESS.run(self.PC, {self.TA: ant})


def main():
    ant, con, one = ReadData('../data/oil_rev.txt')
    skf = KFold(n_splits=10, shuffle=True, random_state=0)
    for train_mask, test_mask in skf.split(ant):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]


if __name__ == "__main__":
    main()
