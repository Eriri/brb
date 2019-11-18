import math
import numpy as np
import tensorflow as tf
from view import *
import time


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
    def __init__(self, rule_num, one, util, raw_x, raw_y):
        self.x = tf.placeholder(tf.float64, shape=[None, 2])
        self.y = tf.placeholder(tf.float64, shape=[None])
        self.a = tf.Variable(raw_x, trainable=True, dtype=tf.float64)
        self.b = tf.Variable(raw_y, trainable=True, dtype=tf.float64)
        self.d = tf.Variable(np.log(one), trainable=True, dtype=tf.float64)
        self.r = tf.Variable(np.random.normal(size=(rule_num,)), trainable=True, dtype=tf.float64)
        # self.u = tf.constant(np.array(util), dtype=tf.float64)
        self.u = tf.Variable(np.array(util), trainable=True, dtype=tf.float64)

        self.w = tf.math.square((self.a - tf.expand_dims(self.x, -2))/tf.math.exp(self.d))
        self.aw = tf.math.exp(-tf.reduce_sum(self.w, -1)) * tf.exp(self.r)
        self.sw = tf.reduce_sum(self.aw, -1)
        self.bc = tf.reduce_prod(tf.expand_dims(self.aw/(tf.expand_dims(self.sw, -1)-self.aw), -1)*tf.nn.softmax(self.b)+1.0, -2)-1.0
        self.pc = self.bc / tf.expand_dims(tf.reduce_sum(self.bc, -1), -1)
        self.o = tf.reduce_sum(self.pc * self.u, -1)

        self.error = tf.reduce_mean(tf.math.abs(self.y - self.o))
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


def main():
    st = time.time()
    x, y = generate()
    one = 0.2 * np.ptp(x, axis=0)
    util = np.array([0, 2, 4, 6, 8])
    mask = np.arange(2007)

    yc = []
    for yi in y:
        z = -10 * np.zeros_like(util)
        for i in range(5):
            if util[i] == yi:
                z[i] = 0.0
        for i in range(4):
            if util[i] < yi and yi < util[i+1]:
                z[i] = max(-10, np.log(util[i+1]-yi))
                z[i+1] = max(-10, np.log(yi-util[i]))
        yc.append(z)
    yc = np.array(yc)

    np.random.shuffle(mask)
    raw_x, raw_y = x[mask[:64]], yc[mask[:64]]
    model = FM(64, one, util, raw_x, raw_y)
    model.train(x, y, 400)
    print(np.mean(np.abs(model.result(x)-y)))
    ed = time.time()
    print("time(s): %f" % (ed-st))


if __name__ == "__main__":
    main()
