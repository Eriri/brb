import tensorflow as tf
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, StratifiedKFold
from view import draw2d, draw2d3


class Model(tf.keras.Model):
    def __init__(self, rule_num, att_dim, res_dim, low, high, one):
        super(Model, self).__init__()

        self.AN = tf.Variable(np.random.uniform(low, high, size=(rule_num, att_dim)), dtype=tf.float64, trainable=True)
        self.D = tf.Variable(np.log(one), dtype=tf.float64, trainable=True)
        self.B = tf.Variable(tf.random.normal(shape=(rule_num, res_dim,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.R = tf.Variable(tf.random.normal(shape=(rule_num,), dtype=tf.float64), dtype=tf.float64, trainable=True)

        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])

    def call(self, inputs):
        w = tf.math.square((self.AN - tf.expand_dims(inputs, -2)) / tf.math.exp(self.D))
        # w = tf.math.square((self.AN - tf.expand_dims(inputs, -2)) / (tf.math.square(self.D)+0.01))
        # aw = tf.exp(-tf.reduce_sum(w, -1)) * (tf.math.square(self.R) + 0.1)
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.R)
        sw = tf.reduce_sum(aw, -1)
        bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(self.B)+1.0, -2)-1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return pc


def main():
    ds = fetch_openml('glass')
    x = np.array(ds['data'], np.float)
    y = np.argwhere(ds['target'][:, None] == np.array(list(set(ds['target']))))[:, 1]
    y = np.array(y, np.float)
    c = len(list(set(ds['target'])))
    # kf = KFold(n_splits=10)
    kf = StratifiedKFold(n_splits=10)
    L = []
    for train_mask, test_mask in kf.split(x, y):
        train_x, train_y = x[train_mask], y[train_mask]
        test_x, test_y = x[test_mask], y[test_mask]
        brb = Model(256, train_x.shape[1], c, np.min(x, 0), np.max(x, 0), np.ptp(x, 0))
        brb.fit(x=train_x, y=train_y, batch_size=8, epochs=64000, verbose=1)
        loss, acc = brb.evaluate(x=test_x, y=test_y)
        L.append(loss)
        t = input()
        print(t)


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    main()
