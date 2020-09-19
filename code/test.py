import tensorflow as tf
import numpy as np
<<<<<<< HEAD
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.keras.backend.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
# tf.debugging.enable_check_numerics()
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()
dtype = tf.float32
eps_mul = tf.constant(1e-3, dtype=dtype)
eps_add = tf.constant(1e-30, dtype=dtype)

=======
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

a = tf.Variable([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
b = tf.constant([[0., np.nan, 0.], [np.nan, np.nan, 2.]])
y = tf.constant([0.,1.])
with tf.GradientTape() as gt:
    c = a - tf.expand_dims(b, -2)
    d = tf.where(tf.math.is_nan(c), tf.zeros_like(c), c)
    aw = tf.exp(-tf.reduce_sum(tf.math.square(d), -1))
    p = tf.reduce_sum(aw, -1)
    error = tf.keras.losses.mse(y,p)
>>>>>>> 1adf3747c1285da99a188dea7ccd06289480272a

class Model(BaseEstimator):
    def __init__(self, rule_num, att_dim, res_dim, batch_size, epoch):
        self.brb = None
        self.rule_num = rule_num
        self.att_dim = att_dim
        self.res_dim = res_dim
        self.batch_size = batch_size
        self.epoch = epoch

<<<<<<< HEAD
    def fit(self, x, y):
        s = tf.distribute.MirroredStrategy()
        ds = tf.data.Dataset.from_tensor_slices((x, tf.one_hot(y, self.res_dim)))
        ds = ds.shuffle(1024).batch(self.batch_size).repeat(self.epoch)
        ds = s.experimental_distribute_dataset(ds)
        with s.scope():
            self.brb = create_brb(self.rule_num, self.att_dim, self.res_dim, x, y)
            opt, tv = tf.optimizers.SGD(), brb.trainable_variables

            def calculate_loss(y_true, y_pred):
                return tf.nn.compute_average_loss(
                    tf.keras.losses.categorical_crossentropy(y_true, y_pred),
                    global_batch_size=bs)

            def train_step(x, y):
                with tf.GradientTape(persistent=True) as gt:
                    loss = calculate_loss(y, model.predict(x))
                opt.apply_gradients(zip(gt.gradient(loss, tv), tv))
                return loss

            @tf.function
            def dist_train_step(x, y):
                per_loss = s.run(train_step, args=(x, y,))
                return s.reduce(tf.distribute.ReduceOp.SUM, per_loss, None)

            for step, (x, y) in enumerate(ds):
                loss = dist_train_step(x, y)
                tf.summary.scalar("loss", loss.numpy(), step)

    def predict(self, x):
        return self.brb.predict(x).numpy()

    def predict_prob(self, x):
        return self.brb.output(x).numpy()


class BRB(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim, raw_x, raw_y):
        super(BRB, self).__init__()
        self.a, self.b = tf.Variable(raw_x, dtype=dtype), tf.Variable(raw_y, dtype=dtype)
        self.d = tf.Variable(tf.ones(shape=(att_dim,), dtype=dtype))
        self.r = tf.Variable(tf.ones(shape=(rule_num,), dtype=dtype))

    def output(self, x):
        w = tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.nn.relu(self.d)
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.nn.relu(self.r)
        bw = aw + eps_mul * tf.expand_dims(tf.reduce_max(aw, -1), -1) + eps_add
        cw = tf.expand_dims(bw / (tf.expand_dims(tf.reduce_sum(bw, -1), -1) - bw), -1)
        bc = tf.reduce_prod(cw * tf.nn.softmax(self.b) + 1.0, -2) - 1.0 + eps_add
        return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)

    def predict(self, x):
        return tf.argmax(self.output(x), -1)


def create_brb(rule_num, att_dim, res_dim, x, y):


if __name__ == "__main__":
    model = Model(16, 64, 1000)
    model.fit([1], [2])
=======
print(gt.gradient(error, a))
>>>>>>> 1adf3747c1285da99a188dea7ccd06289480272a
