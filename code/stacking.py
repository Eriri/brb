import os
import tensorflow as tf
import numpy as np
from util import kfold, generate_variable
from dataset import dataset_numeric_classification
from imblearn.over_sampling import RandomOverSampler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.keras.backend.set_floatx('float64')
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()


class Model(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim):
        super(Model, self).__init__()
        self.a = generate_variable((rule_num, att_dim,))
        self.d = generate_variable((att_dim,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))

    def predict(self, x):
        w = tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d)
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)

<<<<<<< HEAD
    def evaluate(self, x):
        return self.predict(x)[:, 1]


class Modelz(tf.Module):
    def __init__(self, rule_num, att_dim, sub_dim, res_dim):
        super(Modelz, self).__init__()
        self.a = generate_variable((rule_num, att_dim,))
        self.d = generate_variable((att_dim,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))
        # self.s = generate_variable((rule_num, sub_dim,))
        self.s = tf.Variable(tf.random.normal((rule_num, sub_dim,), 0.5, dtype=tf.float64))
        self.z = generate_variable((res_dim,))

    def predict(self, xa, xs):
        wa = tf.math.square(self.a - tf.expand_dims(xa, -2)) * tf.math.exp(self.d)
        ws = tf.math.square(self.s - tf.expand_dims(xs, -2)) * tf.math.exp(self.z)
        aw = tf.exp(-tf.reduce_sum(tf.concat([wa, ws], -1), -1)) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)


def training(rule_num, att_dim, res_dim, x, y, bs=64, ep=1024):
    s = tf.distribute.MirroredStrategy()
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    ds = s.experimental_distribute_dataset(ds)
    with s.scope():
        model = Model(rule_num, att_dim, res_dim)
        opt, tv = tf.optimizers.Adam(), model.trainable_variables

        def calculate_loss(y_true, y_pred):
            return tf.nn.compute_average_loss(
                tf.keras.losses.categorical_crossentropy(y_true, y_pred), global_batch_size=bs)

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


def trainingz(rule_num, att_dim, sub_dim, res_dim, sub_models, x, y, bs=64, ep=1024):
    s = tf.distribute.MirroredStrategy()
=======

class Modelz(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim):
        super(Modelz, self).__init__()
        self.a = generate_variable((rule_num, att_dim,))
        self.d = generate_variable((att_dim,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))
        self.z = generate_variable((rule_num, res_dim,))

    def predict(self, x, zx):
        w = tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d)
        wz = tf.losses.categorical_crossentropy(tf.expand_dims(zx, -2), tf.nn.softmax(self.z))
        aw = tf.exp(-tf.reduce_sum(tf.where(tf.math.is_nan(w), tf.zeros_like(w), w), -1) - wz) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.where(tf.math.is_nan(pc), tf.zeros_like(pc), pc)


def training(model, x, y, bs=32, ep=1024):
>>>>>>> 1adf3747c1285da99a188dea7ccd06289480272a
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    ds = s.experimental_distribute_dataset(ds)
    with s.scope():
        modelz = Modelz(rule_num, att_dim, sub_dim, res_dim)
        opt, tv = tf.optimizers.Adam(), modelz.trainable_variables

        def calculate_loss(y_true, y_pred):
            return tf.nn.compute_average_loss(
                tf.keras.losses.categorical_crossentropy(y_true, y_pred), global_batch_size=bs)

        def train_step(x, y):
            xz = tf.concat([tf.expand_dims(sm.evaluate(x), -1) for sm in sub_models], -1)
            with tf.GradientTape(persistent=True) as gt:
                loss = calculate_loss(y, modelz.predict(x, xz))
            opt.apply_gradients(zip(gt.gradient(loss, tv), tv))
            return loss

        @tf.function
        def dist_train_step(x, y):
            per_loss = s.run(train_step, args=(x, y,))
            return s.reduce(tf.distribute.ReduceOp.SUM, per_loss, None)

        for step, (x, y) in enumerate(ds):
            loss = dist_train_step(x, y)
            tf.summary.scalar("lossz", loss.numpy(), step)
    return modelz


def evaluating(modelz, sub_models, x):
    return tf.argmax(modelz.predict(x, tf.concat([tf.expand_dims(sm.evaluate(x), -1) for sm in sub_models], -1)), -1)


def trainingz(modelz, sub_models, x, y, bs=64, ep=1024):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    opt, tv = tf.optimizers.Adam(), modelz.trainable_variables
    total = ep * np.math.ceil(y.shape[0] / bs)
    for cnt, (x_, y_) in enumerate(ds):
        zx_ = tf.concat([tf.expand_dims(sub_model.predict(x_)[:, 1], -1) for sub_model in sub_models], -1)
        zx_ = zx_ / tf.expand_dims(tf.reduce_sum(zx_, -1), -1)
        with tf.GradientTape(persistent=True) as gt:
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_, modelz.predict(x_, zx_)))
        opt.apply_gradients(zip(gt.gradient(loss, tv), tv))
        # print('%d/%d' % (cnt, total), end='\r')


def evaluate(modelz, sub_models, x):
    zx = tf.concat([tf.expand_dims(sub_model.predict(x)[:, 1], -1) for sub_model in sub_models], -1)
    zx = zx / tf.expand_dims(tf.reduce_sum(zx, -1), -1)
    return tf.math.argmax(modelz.predict(x, zx), -1)


def main():
    experiment_num = 10
<<<<<<< HEAD
    data, target, att_dim, res_dim = dataset_numeric_classification('yeast', 1)
    acc, cnt = tf.metrics.Accuracy(), 0
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric', random_state=en):
            Ot = 0.5 * (np.nanmax(train_data, 0) + np.nanmin(train_data, 0))
            Dt = 0.5 * (np.nanmax(train_data, 0) - np.nanmin(train_data, 0))
            Dt = np.where(Dt == 0.0, 0.5, Dt)
            train_data, test_data = (train_data - Ot) / Dt, (test_data - Ot) / Dt

            sub_models, ros = [], RandomOverSampler(random_state=en)
            for rd in range(res_dim):
                x, y = ros.fit_resample(train_data, train_target == rd)
                sub_models.append(training(16, att_dim, 2, x, np.eye(2)[y.astype(int)]))
            modelz = trainingz(32, att_dim, res_dim, res_dim, sub_models, train_data, np.eye(res_dim)[train_target])

            test_pred = evaluating(modelz, sub_models, test_data)
            acc.update_state(test_pred, test_target)
            tf.summary.scalar('acc', acc.result().numpy(), cnt)
            cnt += 1
=======
    data, target, att_dim, res_dim = dataset_numeric_classification('diabetes', 1)
    acc, acci = tf.metrics.Accuracy(), tf.metrics.Accuracy()
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric', random_state=en):

            O, D = 0.5*(np.min(train_data, 0) + np.max(train_data, 0)), 0.5 * np.ptp(train_data, 0)
            D = np.where(D == 0.0, 0.5, D)
            sub_models = []
            for sub_model in range(res_dim):
                model = Model(32, att_dim, 2)
                training(model, (train_data - O) / D, np.eye(2)[(train_target == sub_model).astype(int)])
                sub_models.append(model)
            mz = Modelz(32, att_dim, res_dim)
            trainingz(mz, sub_models, (train_data - O) / D, np.eye(res_dim)[train_target])

            test_pred = evaluate(mz, sub_models, (test_data - O) / D)

            acci.reset_states()
            acci.update_state(test_target, test_pred)
            acc.update_state(test_target, test_pred)
            print("%lf" % acci.result().numpy())

        print("acc=%lf" % acc.result().numpy())
    os.system('paplay beep.wav')
>>>>>>> 1adf3747c1285da99a188dea7ccd06289480272a


if __name__ == "__main__":
    main()
