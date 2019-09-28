import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from ADASYN import ADASYN
import random


class BRB:
    def __init__(self, one, att_dim, res_dim, base_num, base_ant, base_con):
        self.O = tf.Variable(one, expected_shape=[att_dim])
        self.A, self.C = tf.constant(base_ant), tf.constant(base_con)
        self.TA, self.TC = tf.placeholder(tf.float64, [None, att_dim]), tf.placeholder(tf.float64, [None, res_dim])
        self.RW = tf.Variable(tf.random_normal((base_num,), dtype=tf.float64))
        self.W = tf.exp(-tf.reduce_sum(tf.square((self.A-self.TA[:, None])/self.O), axis=2))*tf.log_sigmoid(self.RW)+tf.cast(1e-8, tf.float64)
        self.SW = tf.reduce_sum(self.W, axis=1)
        self.B = tf.reduce_prod(tf.expand_dims(self.W/(self.SW[:, None]-self.W), -1)*self.C+tf.cast(1, tf.float64), -2)-1
        self.PC = self.B / tf.reduce_sum(self.B, -1)[:, None]
        self.ERROR = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.TC, self.PC))
        self.PRED = tf.equal(tf.argmax(self.TC, -1), tf.argmax(self.PC, -1))
        self.ACC = tf.reduce_mean(tf.cast(self.PRED, tf.float64))
        self.STEP = tf.train.AdamOptimizer().minimize(self.ERROR)

    def train(self, ant, con, ep=10, bs=64):
        cb = int(np.floor(len(ant)/bs))
        with tf.Session() as s:
            s.run(tf.global_variables_initializer())
            for e in range(ep):
                ac = list(zip(ant, con))
                random.shuffle(ac)
                ant, con = zip(*ac)
                for b in range(cb):
                    s.run([self.ERROR, self.STEP], {self.TA: ant[b*bs:(b+1)*bs], self.TC: con[b*bs:(b+1)*bs]})
                    print(e, s.run([self.ACC], {self.TA: ant, self.TC: con}))


def ReadData(filename):
    ant, con = [], []
    with open(file=filename, mode='r') as f:
        for i in f:
            e = list(map(float, i.split()))
            ant.append(e[:-1])
            t = [0.0, 0.0]
            t[int(int(e[-1]) != 1)] = 1.0
            con.append(t)
    one = np.ptp(ant, axis=0)
    return np.array(ant), np.array(con), np.array(one)


def TrainBRB(ant, con, one):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    adsn = ADASYN()
    BS = []
    for train_mask, test_mask in skf.split(ant, np.argmax(con, -1)):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]
        train_ant, train_con_argmax = adsn.fit_transform(train_ant, np.argmax(train_con, -1))
        train_con = np.zeros((train_con_argmax.shape[0], 2,))
        train_con[train_con_argmax == 1.0] += np.array([0.0, 1.0])
        train_con[train_con_argmax == 0.0] += np.array([1.0, 0.0])
        b = BRB(one, ant.shape[-1], con.shape[-1], len(train_ant), train_ant, train_con)
        b.train(test_ant, test_con)
        BS.append(b)


def main():
    ant, con, one = ReadData('../data/page-blocks.data')
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_mask, test_mask in skf.split(ant, np.argmax(con, -1)):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]
        TrainBRB(train_ant, train_con, one)


if __name__ == "__main__":
    main()
