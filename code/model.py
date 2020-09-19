import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import kfold, generate_variable, random_replace_with_nan
from dataset import dataset_numeric_classification, dataset_mammographic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.keras.backend.set_floatx('float32')
tdtype, ndtype = tf.float32, np.float32
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
# tf.debugging.enable_check_numerics()
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()


def model_loss(model):
    @tf.function
    def loss(y_true, y_pred):
        # return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return model.l1(model.c) + model.l2(model.a) + tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return loss


class BRB(tf.keras.Model):
    def __init__(self, rn, ad, rd, rx=None, ry=None):
        super(BRB, self).__init__()
        # [rn, ad]
        if rx is not None:
            self.a = tf.Variable(rx, dtype=tdtype)
        else:
            self.a = tf.Variable(tf.random.normal((rn, ad,), dtype=tdtype))
        # [rn, rd]
        if ry is not None:
            self.b = tf.Variable(ry, dtype=tdtype)
        else:
            self.b = tf.Variable(tf.zeros((rn, rd,), dtype=tdtype))
        self.c = tf.Variable(tf.ones((rn, ad,), tdtype))  # [rn, ad]
        # self.bn = tf.keras.layers.BatchNormalization()
        # self.l1 = tf.keras.regularizers.l1(0.01/(rn * ad))
        # self.l2 = tf.keras.regularizers.l2(0.01/(rn * ad))
        self.eps = tf.constant(1e-10, tdtype)

    def self_compile(self):
        self.compile(optimizer=tf.keras.optimizers.Nadam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def change_ab(self, trainable):
        self.a = tf.Variable(self.a, trainable=trainable)
        self.b = tf.Variable(self.b, trainable=trainable)
        self.compile(optimizer=tf.keras.optimizers.Nadam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def change_c(self, trainable):
        self.c = tf.Variable(self.c, trainable=trainable)
        self.compile(optimizer=tf.keras.optimizers.Nadam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def call(self, x, training):  # [None,ad]
        w = tf.math.square(self.c * (self.a - tf.expand_dims(x, -2)))  # [None, rn, ad]
        aw = tf.math.exp(tf.negative(tf.reduce_sum(w, -1)))  # [None, rn]
        mj, md = tf.expand_dims(aw, -1) * tf.nn.softmax(self.b), 1.0 - aw  # [None, rn, rd], [None, rn]
        bc = tf.reduce_prod(mj + tf.expand_dims(md, -1), -2) - tf.expand_dims(tf.reduce_prod(md), -1) + self.eps  # [None, rd]

        # return tf.nn.softmax(self.bn(bc))

        return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)


def create(rn, ad, rd, x, y):
    mask, rng = np.arange(x.shape[0]), np.random.default_rng()
    np.random.shuffle(mask)
    bx, by = x[mask[:rn]], np.eye(rd)[y[mask[:rn]]].astype(np.bool)
    sn = -np.ones((rn, rd))
    sn[by] = -sn[by]
    return BRB(rn, ad, rd, bx, sn)


def training(x, y, rn, ad, rd, ep, bs):
    s = tf.distribute.MirroredStrategy()
    with s.scope():
        # model = BRB(rn, ad, rd)
        model = create(rn, ad, rd, x, y)
        tb = tf.keras.callbacks.TensorBoard(log_dir='logs')
        model.change_ab(False)
        model.fit(x, y, batch_size=bs, epochs=500, verbose=0, callbacks=[tb])
        model.change_ab(True)
        model.change_c(False)
        model.fit(x, y, batch_size=bs, epochs=500, verbose=0, callbacks=[tb])
        model.change_c(True)
        model.fit(x, y, batch_size=bs, epochs=500, verbose=0, callbacks=[tb])

    return model


'''
32 128 4000 0.75 C   73.52 87.87 81.81 67.64 81.81 73.52 6
32 128 4000 0.5  C   73.52 84.84 81.81 67.64 81.81 64.70 3
32 128 4000 0.25 C   67.64 87.87 78.78 73.52 72.27 76.47 1

7353 8787 7879 6471
'''


def main():
    experiment_num, data_name = 20, "ecoli"
    rule_num, epoch, batch_size = 32, 2000, 128
    data, target, att_dim, res_dim = dataset_numeric_classification(data_name, 1)
    data = StandardScaler().fit_transform(data).astype(ndtype)
    sum_acc, sum_cnt = 0.0, 0
    pmt = [20, 37, 48, 63, 78, 92]
    tmp = []
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric', random_state=en):
            if sum_cnt in pmt:
                model = training(train_data, train_target,
                                 rule_num, att_dim, res_dim,
                                 epoch, batch_size)
                loss, acc = model.evaluate(test_data, test_target)
                sum_acc += acc
                tmp.append(acc)
            sum_cnt += 1
            # tf.summary.scalar('acc_each_%s' % data_name, acc, sum_cnt)
            tf.summary.scalar('acc_%s' % data_name, sum_acc / sum_cnt, sum_cnt)


if __name__ == "__main__":
    main()
