import os
import tensorflow as tf
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from util import kfold, generate_variable
from dataset import dataset_numeric_classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.keras.backend.set_floatx('float64')
# tf.debugging.enable_check_numerics()
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()
dtype = tf.float64
ntype = np.float64
eps_mul = tf.constant(1e-3, dtype=dtype)
eps_add = tf.constant(1e-30, dtype=dtype)


class Model(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim, raw_x, raw_y):
        super(Model, self).__init__()
        self.a = tf.Variable(initial_value=raw_x, dtype=dtype)  # [rule_num,att_dim]
        self.b = tf.Variable(initial_value=raw_y, dtype=dtype)  # [rule_num, res_dim]
        self.d = tf.Variable(initial_value=tf.zeros(shape=(att_dim,), dtype=dtype))
        self.r = tf.Variable(initial_value=tf.ones(shape=(rule_num,), dtype=dtype))

    def predict(self, x):  # [None, att_dim]
        w = tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d)  # [None, rule_ num, att_dim]
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)  # [None, rule_num]
        aw = aw + eps_mul * tf.expand_dims(tf.reduce_max(aw, -1), -1) + eps_add
        aw = aw / tf.expand_dims(tf.reduce_sum(aw, -1), -1)
        mb = tf.expand_dims(aw, -1) * tf.nn.softmax(self.b)  # [None, rule_num, res_dim]
        md = 1.0 - aw  # [None, rule_num]
        bc = tf.reduce_prod(mb + tf.expand_dims(md, -1), -2) - tf.expand_dims(tf.reduce_prod(md, -1), -1)  # [None, res_dim]
        return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)

    def evaluate(self, x):
        return tf.math.argmax(self.predict(x), -1)


def creat_model(rule_num, att_dim, res_dim, x, y, extra=0.0):
    model_x, model_y = [], []
    sample_size = x.shape[0]
    rng = np.random.default_rng()
    y = tf.argmax(y, -1).numpy()
    for i in range(sample_size):
        model_x.append(x[i])
        p = -np.abs(rng.standard_normal((res_dim,)))
        p[y[i]] = -p[y[i]]
        model_y.append(p)
    extra_size = int(sample_size * extra)
    for i in range(extra_size):
        model_x.append(rng.standard_normal((att_dim,)))
        model_y.append(np.zeros((res_dim,)))
    model_x, model_y = np.array(model_x), np.array(model_y)
    mask = np.random.permutation(len(model_x))
    return Model(rule_num, att_dim, res_dim, model_x[mask[:rule_num]], model_y[mask[:rule_num]])


def training(rule_num, att_dim, res_dim, x, y, bs=64, ep=2000):
    s = tf.distribute.MirroredStrategy()
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    ds = s.experimental_distribute_dataset(ds)
    with s.scope():
        # model = creat_model(rule_num, att_dim, res_dim, x, y, extra=0.0)
        model = Model(rule_num, att_dim, res_dim,
                      tf.random.normal((rule_num, att_dim,), dtype=dtype),
                      tf.zeros((rule_num, res_dim,), dtype=dtype))
        opt, tv = tf.optimizers.SGD(), model.trainable_variables

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
    return model


def main():
    experiment_num = 20
    data, target, att_dim, res_dim = dataset_numeric_classification('ecoli', 1)
    # mass = pandas.read_csv('mass.csv').to_numpy()
    # data, target, att_dim, res_dim = mass[:, :5], mass[:, -1], 5, 2

    data = StandardScaler().fit_transform(data).astype(ntype)
    acc, cnt = tf.metrics.Accuracy(), 0
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric', random_state=en):
            model = training(32, att_dim, res_dim, train_data, tf.one_hot(train_target, res_dim))
            acc.update_state(test_target, model.evaluate(test_data))
            tf.summary.scalar('acc', acc.result().numpy(), cnt)
            cnt += 1


if __name__ == "__main__":
    main()
