import tensorflow as tf
import numpy as np
import time
from view import draw2d, draw2d2, drawloss


class Model(tf.keras.Model):
    def __init__(self, rule_num, res_dim, raw_x=None, raw_y=None):
        super(Model, self).__init__()
        if raw_x is None:
            self.a = tf.Variable(tf.random.normal((rule_num,)))
        else:
            self.a = tf.Variable(raw_x)
        if raw_y is None:
            self.b = tf.Variable(tf.random.normal((rule_num, res_dim,)))
        else:
            self.b = tf.Variable(raw_y)
        self.r, self.d = tf.Variable(tf.zeros((rule_num,))), tf.Variable(tf.ones((1,)))
        self.u = tf.constant([-0.5, 0.0, 0.5, 1.0, 1.5])
        self.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9), loss=tf.keras.losses.MSE, metrics=['mse'])

    def call(self, inputs):
        inputs = inputs / 5.0
        w = tf.math.square(tf.expand_dims(self.a, -1) - tf.expand_dims(inputs, -2)) * tf.math.exp(self.d)
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(tf.reduce_sum(aw, -1), -1)-aw), -1)*tf.nn.softmax(self.b)+1.0, -2)-1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.reduce_sum(pc * self.u, -1)

def create_model(rule_num, raw_x, raw_y):
    mask = np.random.permutation(len(raw_x))
    model_x, model_y = raw_x[mask[:rule_num]], raw_y[mask[:rule_num]]
    util = [-0.5, 0.0, 0.5, 1.0, 1.5]
    rng, dist_y = np.random.default_rng(), []
    for yi in model_y:
        yb = -10*np.abs(rng.standard_normal(5))
        for i in range(4):
            if util[i] < yi and yi < util[i+1]:
                yb[i], yb[i+1]=-yb[i],-yb[i+1]
        dist_y.append(yb)
    dist_y = np.array(dist_y)
    return Model(rule_num,5,model_x.astype(np.float32),dist_y.astype(np.float32))


def main():
    def g(x): return np.exp(-np.square(x-2))+0.5*np.exp(-np.square(x+2))
    x = np.linspace(-5, 5, 1000)
    y = g(x)
    # draw2d(x, y, 'x=-5:0.01:5', 'f(x)')
    # brb = create_model(16, x, y)
    brb = Model(16,5)
    # draw2d2(x, y, x, brb.predict(x), 'x=-5:0.01:5', 'y', 'f(x)','ebrb')
    st = time.time()
    history = brb.fit(x=x, y=y, batch_size=64, epochs=2000, verbose=1)
    ed = time.time()
    # loss = np.array(history.history['loss'])
    # pt = np.linspace(0, ed-st, loss.shape[0])
    # draw2d(pt,loss,'second','MSE')
    # drawloss(loss, pt)

    y_ = brb.predict(x=x)
    # draw2d2(x, y, x, y_, 'x=-5:0.01:5', 'y', 'f(x)', 'momentum_brb')
    # draw2d(x, np.square(y-y_), 'x=-5:0.01:5', 'mse')
    print(ed-st, np.mean(np.square(y-y_)))
    # X = brb.AN.numpy()
    # Y = tf.reduce_sum(tf.nn.softmax(brb.B) * brb.U, -1).numpy()
    # draw2d3(x, y, X, Y)


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float32')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    main()
