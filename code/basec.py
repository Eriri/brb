import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import kfold, generate_variable
from dataset import dataset_numeric_classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.keras.backend.set_floatx('float64')
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()
dtype = tf.float32
eps_mul = tf.constant(1e-3, dtype=dtype)
eps_add = tf.constant(1e-30, dtype=dtype)


class Model(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim, raw_x, raw_y):
        super(Model, self).__init__()
        self.a = tf.Variable(initial_value=raw_x,dtype=dtype)
        self.b = tf.Variable(initial_value=raw_y,dtype=dtype)
        self.d = tf.Variable(initial_value=tf.ones(shape=(att_dim,), dtype=dtype))
        self.r = tf.Variable(initial_value=tf.zeros(shape=(rule_num,), dtype=dtype))

    def predict(self, x):
        w = tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d)
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)
        bw = aw + eps_mul * tf.expand_dims(tf.reduce_max(aw, -1), -1) + eps_add
        cw = tf.expand_dims(bw / (tf.expand_dims(tf.reduce_sum(bw, -1), -1) - bw), -1)
        bc = tf.reduce_prod(cw * tf.nn.softmax(self.b) + 1.0, -2) - 1.0 + eps_add
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



def training(rule_num, att_dim, res_dim, x, y, bs=64, ep=6000):
    s = tf.distribute.MirroredStrategy()
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    ds = s.experimental_distribute_dataset(ds)
    with s.scope():
        model = creat_model(rule_num, att_dim, res_dim, x, y, 0.5)
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
    data = StandardScaler().fit_transform(data).astype(np.float32)
    acc, cnt = tf.metrics.Accuracy(), 0
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric', random_state=en):
            model = training(256, att_dim, res_dim, train_data, tf.one_hot(train_target, res_dim))
            acc.update_state(test_target, model.evaluate(test_data))
            tf.summary.scalar('acc', acc.result().numpy(), cnt)
            cnt += 1


if __name__ == "__main__":
    main()
