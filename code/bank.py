import tensorflow as tf
import numpy as np
import pandas as pd
import math
import tqdm
from sklearn.metrics import roc_auc_score


class EBRB:  # metric_shape category_shapes one low high rule_num result_shape
    def __init__(self, ms, cs, on, lo, hi, rn, rs):
        self.xm = tf.placeholder(dtype=tf.float64, shape=[None, ms])
        self.xc = [tf.placeholder(dtype=tf.float64, shape=[None, s]) for s in cs]
        self.xw = tf.placeholder(dtype=tf.float64, shape=[None])
        self.yi = tf.placeholder(dtype=tf.float64, shape=[None, rs])

        self.am = tf.Variable(np.random.uniform(lo, hi, (rn, ms,)), dtype=tf.float64)
        self.ac = [tf.Variable(np.random.normal(0.0, 1.0, (rn, s,)), dtype=tf.float64) for s in cs]
        self.on = tf.Variable(np.log(on), dtype=tf.float64)
        self.co = tf.Variable(np.random.normal(0.0, 1.0, (rn, rs,)), dtype=tf.float64, trainable=False)
        self.rw = tf.Variable(np.random.normal(0.0, 1.0, (rn,)), dtype=tf.float64)

        dac, don, dco, drw = [tf.nn.softmax(a) for a in self.ac], tf.exp(self.on), tf.nn.softmax(self.co), tf.exp(self.rw)

        self.awm = tf.exp(-tf.reduce_sum(tf.square((self.am - tf.expand_dims(self.xm, -2))/don), -1)) * drw
        self.awc = tf.reduce_prod(tf.concat([tf.expand_dims(tf.reduce_sum(tf.sqrt(a * tf.expand_dims(tf.nn.softmax(x), -2)), -1), -1) for a, x in zip(dac, self.xc)], -1), -1)

        self.aw, self.sw = self.awm * self.awc, tf.reduce_sum(self.awm * self.awc, -1)
        self.bc = tf.reduce_prod(tf.expand_dims(self.aw/(tf.expand_dims(self.sw, -1)-self.aw), -1)*dco+1.0, -2)-1.0
        self.yo = self.bc/tf.expand_dims(tf.reduce_sum(self.bc, -1), -1)

        self.err = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.yi, self.yo) * self.xw)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.yi, -1), tf.argmax(self.yo, -1)), tf.float64))

        self.step = tf.train.AdamOptimizer().minimize(self.err)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, mdata, cdata, tdata, dlen, ep=10000, bs=64):
        bn, sh, fd = int(math.ceil(dlen/bs)), np.arange(dlen), dict()
        for e in range(ep):
            np.random.shuffle(sh)
            pb = tqdm.tqdm(total=bn)
            mdata, cdata, tdata = mdata[sh], [c[sh] for c in cdata], tdata[sh]
            for i in range(bn):
                fd[self.xm] = mdata[i*bs:(i+1)*bs]
                for xi, ci in zip(self.xc, cdata):
                    fd[xi] = ci[i*bs:(i+1)*bs]
                fd[self.yi] = tdata[i*bs:(i+1)*bs]
                self.sess.run(self.step, feed_dict=fd)
                pb.update()
            pb.close()
            self.predict(mdata, cdata, tdata)

    def predict(self, mdata, cdata, tdata, bs=1024):
        bn, fd = int(math.ceil(mdata.shape[0]/bs)), {}
        yt, yp = np.argmax(tdata, -1), []
        pb = tqdm.tqdm(total=bn)
        for i in range(bn):
            fd[self.xm] = mdata[i*bs:(i+1)*bs]
            for xi, ci in zip(self.xc, cdata):
                fd[xi] = ci[i*bs:(i+1)*bs]
            yp.append(np.array(self.sess.run(self.pre, fd)))
            pb.update()
        pb.close()
        yp = np.concatenate(yp)
        print(roc_auc_score(yt, yp))


def main():
    mdata = np.load('../data/bank/np/train_mdata.npz')['arr_0']
    one, low, high = np.ptp(mdata, axis=0), np.min(mdata, axis=0), np.max(mdata, axis=0)
    catt = ['loanProduct', 'gender', 'edu', 'job', 'basicLevel', 'ethnic', 'highestEdu', 'linkRela']
    cdata = [np.load('../data/bank/np/train_cdata_%s.npz' % c)['arr_0'] for c in catt]
    cshape = [c.shape[1] for c in cdata]
    tdata = np.load('../data/bank/np/train_tdata.npz')['arr_0']

    model = EBRB(mdata.shape[1], cshape, one, low, high, 100, tdata.shape[1])
    model.train(mdata, cdata, tdata, mdata.shape[0])


if __name__ == "__main__":
    main()
