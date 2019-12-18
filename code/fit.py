import tensorflow as tf
import numpy as np
import time
from view import draw2d, draw2d3


class Model(tf.keras.Model):
    def __init__(self, rule_num, res_dim, low, high, one, util):
        super(Model, self).__init__()
        self.AN = tf.Variable(np.random.uniform(low, high, size=(rule_num,)), dtype=tf.float64, trainable=True)
        self.D = tf.Variable(np.sqrt(one), dtype=tf.float64, trainable=True)
        self.B = tf.Variable(tf.random.normal(shape=(rule_num, res_dim,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.R = tf.Variable(tf.random.normal(shape=(rule_num,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.U = tf.Variable(util, dtype=tf.float64, trainable=False)
        self.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MSE, metrics=['mse'])

    def call(self, inputs):
        w = tf.math.square((tf.expand_dims(self.AN, -1) - tf.expand_dims(inputs, -2)) / (tf.math.square(self.D)+0.1))
        aw = tf.exp(-tf.reduce_sum(w, -1)) * (tf.math.square(self.R) + 0.1)
        # aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.R)
        sw = tf.reduce_sum(aw, -1)
        bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(self.B)+1.0, -2)-1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.reduce_sum(pc * self.U, -1)


def main():
    def f(x): return np.exp(-np.square(x-2))+0.5*np.exp(-np.square(x+2))
    x = np.linspace(-5, 5, 1000)
    y = f(x)
    brb = Model(5, 5, -5, 5, 10, np.array([-0.5, 0.0, 0.5, 1.0, 1.5]))
    st = time.time()
    brb.fit(x=x, y=y, batch_size=32, epochs=1600, verbose=1)
    ed = time.time()
    y_ = brb.predict(x=x)
    draw2d(x, np.square(y-y_))
    print(ed-st, np.mean(np.square(y-y_)))
    X = brb.AN.numpy()
    Y = tf.reduce_sum(tf.nn.softmax(brb.B) * brb.U, -1).numpy()
    draw2d3(x, y, X, Y)


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    main()
