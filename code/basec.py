import os
import tensorflow as tf
import numpy as np
from util import kfold, generate_variable
from dataset import dataset_numeric_classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Model(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim):
        super(Model, self).__init__()
        self.a = generate_variable((rule_num, att_dim,))
        self.d = generate_variable((att_dim,))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))

    def predict(self, x):
        w = tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d)
        aw = tf.exp(-tf.reduce_sum(tf.where(tf.math.is_nan(w), tf.zeros_like(w), w), -1)) * tf.math.exp(self.r)
        bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * tf.nn.softmax(self.b) + 1.0, -2) - 1.0
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
        return tf.where(tf.math.is_nan(pc), tf.zeros_like(pc), pc)

    def evaluate(self, x):
        return tf.math.argmax(self.predict(x), -1)


def training(model, x, y, bs=64, ep=1024):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    opt, tv = tf.optimizers.Adam(), model.trainable_variables
    total = ep * np.math.ceil(y.shape[0] / bs)
    sw = tf.summary.create_file_writer('logs')
    acc = tf.metrics.Accuracy()
    for cnt, (x_, y_) in enumerate(ds):
        with tf.GradientTape(persistent=True) as gt:
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_, model.predict(x_)))
        opt.apply_gradients(zip(gt.gradient(loss, tv), tv))
        acc.reset_states(), acc.update_state(tf.argmax(y, -1), model.evaluate(x))
        with sw.as_default():
            tf.summary.scalar('loss1664', loss.numpy(), step=cnt)
            tf.summary.scalar('acc1664', acc.result().numpy(), step=cnt)
        print('%d/%d' % (cnt, total), end='\r')


def main():
    experiment_num = 10
    data, target, att_dim, res_dim = dataset_numeric_classification('seeds')
    acc, acci = tf.metrics.Accuracy(), tf.metrics.Accuracy()
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric'):
            model = Model(16, att_dim, res_dim)
            O, D = 0.5*(np.min(train_data, 0) + np.max(train_data, 0)), 0.5 * np.ptp(train_data, 0)
            # D = np.where(D == 0.0, 0.5, D)  ecoli
            training(model, (train_data - O) / D, np.eye(res_dim)[train_target])
            test_pred = model.evaluate((test_data - O) / D)
            acci.reset_states()
            acci.update_state(test_target, test_pred)
            acc.update_state(test_target, test_pred)
            print("%lf" % acci.result().numpy())
            exit(0)
        print("acc=%lf" % acc.result().numpy())
    os.system('paplay beep.wav')


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    main()
