import tensorflow as tf
import numpy as np
from util import read_oil, BaseBRB, RIMER
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from tensorflow.keras.models import load_model
import time
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model(tf.keras.Model):
    def __init__(self, rule_num, att_dim, res_dim, low, high, one, util):
        super(Model, self).__init__()
        self.BRB = (  # AN,AC,E,D,B,R
            tf.Variable(np.random.uniform(low, high, size=(rule_num, att_dim,)), dtype=tf.float64, trainable=True),
            None,
            tf.Variable(tf.zeros(shape=(att_dim,), dtype=tf.float64), dtype=tf.float64, trainable=True),
            tf.Variable(np.sqrt(one), dtype=tf.float64, trainable=True),
            tf.Variable(tf.random.normal(shape=(rule_num, res_dim,), dtype=tf.float64), dtype=tf.float64, trainable=True),
            tf.Variable(tf.ones(shape=(rule_num,), dtype=tf.float64), dtype=tf.float64, trainable=True))

        self.U = tf.Variable(util, dtype=tf.float64, trainable=True)
        self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MAE, metrics=['mae'])

    def call(self, inputs):
        return tf.reduce_sum(RIMER((inputs, None), self.BRB, 'con', 'nume') * self.U, -1)


def experiment(x, y, e):
    x, y = shuffle(x, y)
    tx, ty, util = x[:512], y[:512], np.arange(0, 10, 2, np.float64)
    strategy = tf.distribute.MirroredStrategy()
    z = []
    for base in range(10):
        with strategy.scope():
            brb = Model(128, 2, 5, np.min(tx, axis=0), np.max(tx, axis=0), np.ptp(tx, axis=0), util)
        res = []
        for i in range(16):
            st = time.time()
            brb.fit(x=tx, y=ty, batch_size=64*4, epochs=1600, verbose=1)
            py = brb.predict(tx)
            v = [brb.evaluate(x, y, verbose=0)[0], mean_absolute_error(ty, py), r2_score(ty, py), pearsonr(ty, py)[0]]
            ed = time.time()
            print(i, ed-st, v), res.append(v)
            time.sleep(1.0)
        z.append(res), brb.save('models/base_%d_%d' % (e, base))
    return z


def main():
    e = int(sys.argv[1])
    x, y = read_oil()
    res = experiment(x, y, e)
    np.save('eval%d.npy' % e, res)
    print(res)


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    main()
