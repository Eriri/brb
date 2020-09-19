import os
import tensorflow as tf
import numpy as np
from dataset import dataset_numeric_classification
from util import kfold, generate_variable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.keras.backend.set_floatx('float64')
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
sw = tf.summary.create_file_writer('logs')


class BModel(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim):
        super(BModel, self).__init__()
        self.a = generate_variable((rule_num, att_dim,))
        self.d = generate_variable((att_dim,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))

    def __call__(self, x):
        w = tf.math.exp(tf.math.negative(
            tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d)))
        aw = tf.reduce_sum(tf.where(tf.math.is_nan(w), tf.zeros_like(w), w), -1) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.where(tf.math.is_nan(pc), tf.zeros_like(pc), pc)


class AModel(tf.Module):
    def __init__(self, rule_num, base_num, base_rule_num, att_dim, mid_dim, res_dim):
        super(AModel, self).__init__()
        self.bms = [BModel(base_rule_num, att_dim, mid_dim) for _ in range(base_num)]
        self.a = generate_variable((rule_num, base_num, mid_dim,))
        self.d = generate_variable((base_num,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))

    def __call__(self, x):
        mid_out = tf.concat([tf.expand_dims(bm(x), -2) for bm in self.bms], -2)
        w = tf.reduce_sum(tf.math.square(self.a - tf.expand_dims(mid_out, -3)), -1) * tf.math.exp(self.d)
        aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.where(tf.math.is_nan(pc), tf.zeros_like(pc), pc)


def training(rule_num, base_num, base_rule_num, att_dim, mid_dim, res_dim, x, y, bs=64, ep=1024):
    s = tf.distribute.MirroredStrategy()
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    ds = s.experimental_distribute_dataset(ds)
    with s.scope():
        model = AModel(rule_num, base_num, base_rule_num, att_dim, mid_dim, res_dim)
        opt, tv = tf.optimizers.Adam(), model.trainable_variables

        def calculate_loss(y_true, y_pred):
            return tf.nn.compute_average_loss(
                tf.keras.losses.categorical_crossentropy(
                    y_true, y_pred),
                global_batch_size=bs)

        def train_step(x, y):
            with tf.GradientTape(persistent=True) as gt:
                loss = calculate_loss(y, model(x))
            opt.apply_gradients(zip(gt.gradient(loss, tv), tv))
            return loss

        @tf.function
        def dist_train_step(x, y):
            per_loss = s.run(train_step, args=(x, y,))
            return s.reduce(tf.distribute.ReduceOp.SUM, per_loss, None)

        for step, (x, y) in enumerate(ds):
            loss = dist_train_step(x, y)
            with sw.as_default():
                tf.summary.scalar("dist_loss", loss.numpy(), step)
    return model


def main():
    experiment_num = 10
    data, target, att_dim, res_dim = dataset_numeric_classification('ecoli', 1)
    acc = tf.metrics.Accuracy()
    cnt = 0
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric', random_state=en):
            O, D = 0.5*(np.min(train_data, 0) + np.max(train_data, 0)), 0.5 * np.ptp(train_data, 0)
            D = np.where(D == 0.0, 0.5, D)
            model = training(32, 4, 16, att_dim, 4, res_dim, (train_data - O) / D, np.eye(res_dim)[train_target])
            test_pred = tf.math.argmax(model((test_data - O) / D), -1)
            acc.update_state(test_target, test_pred)
            with sw.as_default():
                tf.summary.scalar('ecoli_16*4_32_64', acc.result().numpy(), cnt)
                cnt += 1


if __name__ == "__main__":
    main()
