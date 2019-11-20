import tensorflow as tf
import numpy as np
from util import read_oil


class BRB(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim, low, high, one, util):
        super(BRB, self).__init__(name=None)
        self.A = tf.Variable(np.random.uniform(low, high, size=(rule_num, att_dim,)), dtype=tf.float64, trainable=True)
        self.D = tf.Variable(np.log(one), dtype=tf.float64, trainable=True)
        self.B = tf.Variable(tf.random.normal(shape=(rule_num, res_dim,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.R = tf.Variable(tf.zeros(shape=(rule_num,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.U = tf.Variable(util, dtype=tf.float64, trainable=True)

    def __call__(self, x):
        w = tf.math.square((self.A - tf.expand_dims(x, -2)) / tf.math.exp(self.D))
        aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.R)
        sw = tf.reduce_sum(aw, -1)
        bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(self.B)+1.0, -2)-1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        out = tf.reduce_sum(pc*self.U, -1)
        return out


def train(brb, x, y, ep, bs):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(1024).batch(bs).repeat(ep)
    opt = tf.keras.optimizers.Adam()
    tape = tf.GradientTape()
    for tx, ty in ds:
        with tape:
            # loss = tf.keras.losses.MSE(ty, brb(tx))
            loss = tf.keras.losses.MAE(ty, brb(tx))
        grads = tape.gradient(loss, brb.trainable_variables)
        opt.apply_gradients(zip(grads, brb.trainable_variables))
        print(loss.numpy())


def main():
    x, y = read_oil()
    low, high, one = np.min(x, 0), np.max(x, 0), np.ptp(x, 0)
    util = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    brb = BRB(64, 2, 5, low, high, one, util)
    train(brb, x[-512:], y[-512:], 100, 64)


if __name__ == "__main__":
    main()
