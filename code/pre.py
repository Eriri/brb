import math
import numpy as np
import tensorflow as tf
from view import *
import time
from scipy.stats import pearsonr


def generate():
    x, y = [], []
    with open('../data/oil.data') as f:
        for i in f:
            z = list(map(float, i.split()))
            x.append(z[:2]), y.append(z[-1])
    return np.array(x), np.array(y)
    # x = [0.003*i for i in range(1000)]
    # y = list(map(lambda t: t*math.cos(t*t)-math.sin(t*t), x))
    # y = list(map(lambda t: math.exp(-(t-2)*(t-2))+0.5*math.exp(-(t+2)*(t+2)), x))
    # return np.array(x), np.array(y)


class PM:
    def __init__(self, rule_num, res_dim, low, high, one, util):
        self.x = tf.placeholder(dtype=tf.float64, shape=[None])
        self.y = tf.placeholder(dtype=tf.float64, shape=[None])

        self.a = tf.Variable(np.random.uniform(low, high, (rule_num,)), trainable=True, dtype=tf.float64)
        self.d = tf.Variable(np.log(one), trainable=True, dtype=tf.float64)
        self.b = tf.Variable(np.random.normal(size=(rule_num, res_dim,)), trainable=True, dtype=tf.float64)
        self.r = tf.Variable(np.random.normal(size=(rule_num,)), trainable=True, dtype=tf.float64)
        # self.u = tf.constant(np.array(util), dtype=tf.float64)
        self.u = tf.Variable(np.array(util), trainable=True, dtype=tf.float64)

        # self.dd = tf.math.square((self.a - tf.expand_dims(self.x, -1))/tf.math.exp(self.d))

        self.w = tf.math.square((self.a - tf.expand_dims(self.x, -1))/tf.math.exp(self.d))
        self.aw = tf.math.exp(tf.negative(self.w)) * tf.exp(self.r)
        self.sw = tf.reduce_sum(self.aw, -1)
        self.bc = tf.reduce_prod(tf.expand_dims(self.aw/(tf.expand_dims(self.sw, -1)-self.aw), -1)*tf.nn.softmax(self.b)+1.0, -2)-1.0
        self.pc = self.bc / tf.expand_dims(tf.reduce_sum(self.bc, -1), -1)
        self.o = tf.reduce_sum(self.pc * self.u, -1)

        self.error = tf.reduce_mean(tf.math.square(self.y - self.o))
        self.step = tf.train.AdamOptimizer().minimize(self.error)

        self.SESS = tf.Session()
        self.SESS.run(tf.global_variables_initializer())

    def train(self, x, y, ep=10000, bs=64):
        bn, mask = int(math.ceil(len(x)/bs)), np.arange(len(x))
        for e in range(ep):
            np.random.shuffle(mask)
            tx, ty = x[mask], y[mask]
            for i in range(bn):
                self.SESS.run(self.step, feed_dict={self.x: tx[i*bs:(i+1)*bs], self.y: ty[i*bs:(i+1)*bs]})
            print(e, self.predict(x, y))

    def predict(self, x, y):
        return self.SESS.run(self.error, feed_dict={self.x: x, self.y: y})

    def result(self, x):
        return self.SESS.run(self.o, feed_dict={self.x: x})

    def debug(self, x, y):
        print(self.SESS.run(tf.shape(self.bc), feed_dict={self.x: x, self.y: y}))


class FM:
    def __init__(self, rule_num, low, high, one, util, mi):
        self.x = tf.placeholder(tf.float64, shape=[None, 2])
        self.y = tf.placeholder(tf.float64, shape=[None])
        self.a = tf.Variable(np.random.uniform(low, high, size=(rule_num, 2)), trainable=True, dtype=tf.float64)
        self.b = tf.Variable(np.random.normal(size=(rule_num, 5)), trainable=True, dtype=tf.float64)
        self.d = tf.Variable(np.log(one), trainable=True, dtype=tf.float64)
        self.r = tf.Variable(np.zeros(shape=(rule_num,)), trainable=True, dtype=tf.float64)
        # self.u = tf.constant(np.array(util), dtype=tf.float64)
        self.u = tf.Variable(np.array(util), trainable=True, dtype=tf.float64)

        self.w = tf.math.square((self.a - tf.expand_dims(self.x, -2))/tf.math.exp(self.d))
        self.aw = tf.math.exp(-tf.reduce_sum(self.w, -1)) * tf.exp(self.r)
        self.sw = tf.reduce_sum(self.aw, -1)
        self.bc = tf.reduce_prod(tf.expand_dims(self.aw/(tf.expand_dims(self.sw, -1)-self.aw), -1)*tf.nn.softmax(self.b)+1.0, -2)-1.0
        self.pc = self.bc / tf.expand_dims(tf.reduce_sum(self.bc, -1), -1)
        self.o = tf.reduce_sum(self.pc * self.u, -1)

        self.mae = tf.reduce_mean(tf.math.abs(self.y-self.o))
        self.mse = tf.reduce_mean(tf.math.square(self.y-self.o))
        self.rmse = tf.math.sqrt(self.mse)
        self.error = tf.reduce_sum(tf.math.square(self.y - self.o))
        self.step = tf.train.AdamOptimizer().minimize(self.error)

        self.SESS = tf.Session()
        self.SAVER = tf.train.Saver()
        self.PATH = './model_%d' % mi
        self.SESS.run(tf.global_variables_initializer())

    def train(self, x, y, ep=10000, bs=64):
        bn, mask = int(math.ceil(len(x)/bs)), np.arange(len(x))
        for e in range(ep):
            np.random.shuffle(mask)
            for i in range(bn):
                fd = {self.x: x[mask[i*bs:(i+1)*bs]], self.y: y[mask[i*bs:(i+1)*bs]]}
                self.SESS.run(self.step, feed_dict=fd)
            self.predict(e, x, y)

    def predict(self, e, x, y):
        print(e, self.SESS.run([self.error, self.mae, self.mse], feed_dict={self.x: x, self.y: y}))

    def result(self, x):
        return self.SESS.run(self.o, feed_dict={self.x: x})

    def save(self):
        self.SAVER.save(self.SESS, self.PATH)

    def load(self):
        self.SAVER.restore(self.SESS, self.PATH)


def main():
    st = time.time()
    x, y = generate()
    low, high, one = np.min(x, 0), np.max(x, 0), np.ptp(x, axis=0)
    util = np.array([0, 2, 4, 6, 8])
    mask = np.arange(2007)

    rule_num = 64
    ms, ps = [], []
    for mi in range(25):
        np.random.shuffle(mask)
        model = FM(rule_num, low, high, one, util, mi)
        tx, ty = x[mask[:512]], y[mask[:512]]
        model.train(tx, ty, 6400, 64)
        ms.append(model), ps.append(pearsonr(ty, model.result(tx))[0])
    print(ps)
    for i in np.argsort(np.negative(np.array(ps)))[:5]:
        print(i, ps[i])
        ms[i].save()

    ed = time.time()
    print("time(s): %f" % (ed-st))


if __name__ == "__main__":
    main()
