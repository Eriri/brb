import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import kfold, generate_variable, random_replace_with_nan
from dataset import dataset_numeric_classification

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


class Model(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim, raw_x, raw_y):
        super(Model, self).__init__()
        self.a, self.b = tf.Variable(raw_x, dtype=dtype), tf.Variable(raw_y, dtype=dtype)
        self.d = tf.Variable(tf.ones(shape=(att_dim,), dtype=dtype))
        self.r = tf.Variable(tf.ones(shape=(rule_num,), dtype=dtype))

    def predict(self, x):
        ax = self.a - tf.expand_dims(x, -2)
        ax_ok = tf.where(tf.math.is_nan(ax), tf.zeros_like(ax), ax)
        w = tf.math.square(ax_ok) * tf.math.exp(self.d)
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)
        bw = aw + eps_mul * tf.expand_dims(tf.reduce_max(aw, -1), -1) + eps_add
        cw = tf.expand_dims(bw / (tf.expand_dims(tf.reduce_sum(bw, -1), -1) - bw), -1)
        bc = tf.reduce_prod(cw * tf.nn.softmax(self.b) + 1.0, -2) - 1.0 + eps_add
        return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)

    def evaluate(self, x):
        return tf.math.argmax(self.predict(x), -1)


def creat_model(rule_num, att_dim, res_dim, x=None, y=None):
    if x is None and y is None:
        raw_x = tf.random.normal((rule_num, att_dim,))
        raw_y = tf.random.normal((rule_num, res_dim,))
        return Model(rule_num, att_dim, res_dim, raw_x, raw_y)
    raw_x = [x[(tf.argmax(y, -1) == _).numpy()] for _ in range(res_dim)]
    model_x, model_y = [], []
    idx, rng = 0, np.random.default_rng()
    while len(model_x) < rule_num:
        if len(raw_x[idx]) > 0 and rng.integers(2) == 0:
            xi = raw_x[idx][rng.integers(len(raw_x[idx]))]
            yi = -np.abs(rng.standard_normal((res_dim,)))
            yi[idx] = -yi[idx]
        else:
            xi = rng.standard_normal((att_dim,))
            yi = np.zeros((res_dim,))
        model_x.append(xi), model_y.append(yi)
        idx = (idx + 1) % res_dim
    return Model(rule_num, att_dim, res_dim, model_x, model_y)


def training(rule_num, att_dim, res_dim, x, y, bs=128, ep=2000, missing_rate=0.5):
    s = tf.distribute.MirroredStrategy()
    # ds = tf.data.Dataset.from_tensor_slices((random_replace_with_nan(x, missing_rate).astype(np.float32), y)).shuffle(1024).batch(bs).repeat(ep)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    ds = s.experimental_distribute_dataset(ds)
    with s.scope():
        model = creat_model(rule_num, att_dim, res_dim, x, y)
        opt, tv = tf.optimizers.Adam(), model.trainable_variables

        def calculate_loss(y_true, y_pred):
            return tf.nn.compute_average_loss(
                tf.keras.losses.categorical_crossentropy(y_true, y_pred),
                global_batch_size=bs)

        def train_step(x, y):
            x = tf.where(tf.random.uniform(x.shape, 0., 1.0) > missing_rate, x, np.nan)
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
    return model


def main():
    experiment_num, rule_num, data_name = 50, 32, 'thyroid-new'
    data, target, att_dim, res_dim = dataset_numeric_classification(data_name, 1)
    data = StandardScaler().fit_transform(data).astype(np.float32)
    acc, cnt = tf.metrics.Accuracy(), 0
    missing_rate = 0.5
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 5, 'numeric', en):
            model = training(rule_num, att_dim, res_dim, train_data, tf.one_hot(train_target, res_dim), missing_rate=missing_rate)
            acc.update_state(test_target, model.evaluate(random_replace_with_nan(test_data, missing_rate).astype(np.float32)))
            tf.summary.scalar('acc_%s' % data_name, acc.result().numpy(), cnt)
            cnt += 1


if __name__ == "__main__":
    main()
