import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import kfold, generate_variable, random_replace_with_nan
from dataset import dataset_oil

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')
tdtype, ndtype = tf.float64, np.float64
# tf.debugging.enable_check_numerics()
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()


class BRB(tf.keras.Model):
    def __init__(self, rn, ad, rd, xa, yb):
        super(BRB, self).__init__()
        self.a = tf.Variable(xa, dtype=tdtype)  # [rn, ad]
        self.b = tf.Variable(yb, dtype=tdtype)  # [rn, rd]
        self.c = tf.Variable(0.5 * tf.ones((ad,), dtype=tdtype))
        # self.c = tf.Variable(0.5 * tf.ones((rn, ad,), dtype=tdtype))  # [rn, ad]
        self.u = tf.constant([0.0, 2.0, 4.0, 6.0, 8.0], dtype=tdtype)
        self.eps = tf.constant(1e-10, dtype=tdtype)
        self.self_compile()

    def self_compile(self):
        self.compile(optimizer=tf.keras.optimizers.Nadam(),
                     loss='mse',
                     metrics=['mae', 'mse'])

    def call(self, x, training):  # [None,ad]
        w = tf.math.square(self.c * (self.a - tf.expand_dims(x, -2)))  # [None, rn, ad]
        aw = tf.math.exp(tf.negative(tf.reduce_sum(w, -1)))  # [None, rn]
        aw = aw / tf.expand_dims(tf.reduce_sum(aw, -1), -1)
        mj, md = tf.expand_dims(aw, -1) * tf.nn.softmax(self.b), 1.0 - aw  # [None, rn, rd], [None, rn]
        bc = tf.reduce_prod(mj + tf.expand_dims(md, -1), -2) - tf.expand_dims(tf.reduce_prod(md), -1) + self.eps  # [None, rd]
        pc = bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)  # [None, rd]
        return tf.reduce_sum(pc * self.u, -1)  # [None]


def create(rn, ad, rd, x, y):
    mask = np.argsort(y)
    mask = np.concatenate((mask[:int(rn/2)], mask[-int(rn/2):]))
    return BRB(rn, ad, rd, x[mask],
               [[1, -1, -1, -1, -1]] * int(rn/2) + [[-1, -1, -1, -1, 1]] * int(rn/2))


def training(x, y, rn, ad, rd, ep, bs):
    s = tf.distribute.MirroredStrategy()
    with s.scope():
        # model = BRB(rn, ad, rd, tf.random.normal((rn, ad)), tf.zeros((rn, rd)))
        model = create(rn, ad, rd, x, y)
        tb = tf.keras.callbacks.TensorBoard(log_dir='logs')
        model.fit(x, y, batch_size=bs, epochs=ep, verbose=0, callbacks=[tb])
    return model

# good  16 5000 64 3.417
# bad   16 5000 64


def main():
    data, target = dataset_oil()
    data, target = StandardScaler().fit_transform(data).astype(ndtype), target.astype(ndtype)
    data_name, att_dim, res_dim = "oil", 2, 5
    experiment_num, rule_num, epoch, batch_size = 20, 16, 5000, 64
    sum_mse, sum_cnt = 0.0, 0
    for en in range(experiment_num):
        for _, _, train_data, train_target in kfold(data, target, 4, 'numeric', random_state=en):
            model = training(train_data, train_target,
                             rule_num, att_dim, res_dim,
                             epoch, batch_size)
            loss, mae, mse = model.evaluate(data, target)
            sum_mse += mse
            sum_cnt += 1
            tf.summary.scalar('mse_each_%s' % data_name, mse, sum_cnt)
            tf.summary.scalar('mse_%s' % data_name, sum_mse / sum_cnt, sum_cnt)


if __name__ == "__main__":
    main()
