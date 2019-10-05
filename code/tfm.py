import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import KFold, StratifiedKFold


class EBRB():
    def __init__(self, one, low, high, adim, rdim, rule_num, X):  # (None, adim)
        self.A = tf.Variable(np.random.uniform(low, high, (rule_num, adim,)), dtype=tf.float64, trainable=True)
        self.C = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, rdim,)), dtype=tf.float64, trainable=True)
        self.D = tf.Variable(np.log(one), dtype=tf.float64, trainable=True)
        self.W = tf.Variable(np.random.normal(0.0, 1.0, (rule_num,)), dtype=tf.float64, trainable=True)

        self.BC, self.RW = tf.nn.softmax(self.C), tf.exp(self.W)

        self.AW = tf.exp(-tf.reduce_sum(tf.math.square((self.A-tf.expand_dims(X, -2))/self.D), -1)) * self.RW
        # (rule_num,adim) - (None,1,adim) = (None,rule_num,adim)
        # reduce_sum((None,rule_num,adim),-1) = (None,rule_num)

        self.SW = tf.reduce_sum(self.AW, -1)
        # reduce_sum((None,rule_num),-1) = (None)

        self.B = tf.reduce_prod(tf.expand_dims(self.AW/(tf.expand_dims(self.SW, -1)-self.AW), -1)*self.BC+1.0, -2)-1.0
        # (None,rule_num,1) * (rule_num,rdim) = (None,rule_num,rdim)
        # reduce_prod((None,rule_num,rdim),-2) = (None,rdim)

        self.Y = self.B/tf.expand_dims(tf.reduce_sum(self.B, -1), -1)
        # (None,rdim) / (None,1) = (None,rdim)


class DBRB():
    def __init__(self, anum, adim, rdim, rule_num, X):  # (None, anum, adim)
        self.A = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, anum, adim,)), dtype=tf.float64, trainable=True)
        self.C = tf.Variable(np.random.normal(0.0, 1.0, (rule_num, rdim,)), dtype=tf.float64, trainable=True)
        self.W = tf.Variable(np.random.normal(0.0, 1.0, (rule_num,)), dtype=tf.float64, trainable=True)

        self.DA, self.BC, self.RW = tf.nn.softmax(self.A), tf.nn.softmax(self.C), tf.exp(self.W)

        # t = self.DA * tf.expand_dims(X, -3)
        # t = tf.sqrt(t)
        self.AW = tf.reduce_max(tf.reduce_sum(tf.sqrt(self.DA * tf.expand_dims(X, -3))), -1) * self.RW
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
    def __init__(self, one, low, high, dim_1, dim_2, dim_3, e_num, e_rn, d_rn):
        self.X, self.Y = tf.placeholder(tf.float64, [None, dim_1]), tf.placeholder(tf.float64, [None, dim_3])
        self.E = [EBRB(one, low, high, dim_1, dim_2, e_rn, self.X) for i in range(e_num)]
        self.EO = tf.concat([tf.expand_dims(e.Y, 0) for e in self.E], 0)
        self.D = DBRB(e_num, dim_2, dim_3, d_rn, self.EO)
        self.ERROR = tf.keras.losses.categorical_crossentropy(self.Y, self.D.Y)
        self.STEP = tf.train.AdamOptimizer().minimize(self.ERROR)
        self.ACC = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.Y, -1), tf.argmax(self.D.Y, -1)), tf.float64))
        self.SESS = tf.InteractiveSession()
        self.SESS.run(tf.global_variables_initializer())

    def train(self, oa, oc, ta, tc, ep=10000, bs=64):
        cb = int(len(oa)/bs)
        for e in range(ep):
            oac = list(zip(oa, oc))
            random.shuffle(oac)
            oa, oc = zip(*oac)
            for b in range(cb):
                self.SESS.run(self.STEP, {self.X: oa[b*bs:(b+1)*bs], self.Y: oc[b*bs:(b+1)*bs]})
            print(e, self.SESS.run(self.ACC, {self.X: ta, self.Y: tc}))

    def predict(self, a, c):
        return self.SESS.run(self.ACC, {self.X: a, self.Y: c})


def READ(filename):
    ant, con = [], []
    with open(filename, 'r') as f:
        for i in f:
            e = list(map(float, i.strip().split(',')))
            ant.append(e[1:-1])
            con.append(e[-1])
    les = list(set(con))
    for i in range(len(con)):
        t = les.index(con[i])
        con[i] = np.zeros((len(les),))
        con[i][t] = 1.0
    ant, con = np.array(ant), np.array(con)
    one, low, high = np.ptp(ant, axis=0), np.min(ant, axis=0), np.max(ant, axis=0)
    return ant, con, one, low, high


def main():
    ant, con, one, low, high = READ('../data/glass.data')
    B = BRB(one, low, high, ant.shape[1], 5, con.shape[1], 8, 16, 16)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_mask, test_mask in skf.split(ant, np.argmax(con, -1)):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]
        B.train(train_ant, train_con, test_ant, test_con)
        print(B.predict(test_ant, test_con))
        break


if __name__ == "__main__":
    main()
