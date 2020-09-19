import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import kfold, generate_variable, random_replace_with_nan
from dataset import dataset_numeric_classification, dataset_mammographic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float32')
tdtype, ndtype = tf.float32, np.float32
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
# tf.debugging.enable_check_numerics()
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()


class att_layer(tf.keras.layers.Layer):
    def __init__(self, att_init):
        super(att_layer, self).__init__()
        self.att = tf.Variable(initial_value=att_init, dtype=tdtype)  # [rule_num, att_dim]

    def call(self, inputs):  # [None, att_dim]
        return tf.math.square(self.att - tf.expand_dims(inputs, -2))  # [None, rule_num, att_dim]


class dis_layer(tf.keras.layers.Layer):
    def __init__(self, dis_init):
        super(dis_layer, self).__init__()
        self.dis = tf.Variable(initial_value=dis_init, dtype=tdtype)  # [rule_num, att_dim]

    def call(self, inputs):  # [None, rule_num, att_dim]
        return tf.math.exp(tf.negative(tf.math.reduce_sum(inputs * tf.math.square(self.dis), -1)))  # [None, rule_num]


class res_layer(tf.keras.layers.Layer):
    def __init__(self, res_init):
        super(res_layer, self).__init__()
        self.res = tf.Variable(initial_value=res_init, dtype=tdtype)  # [rule_num, res_dim, 2]
        self.eps = tf.constant(1e-10, dtype=tdtype)

    def call(self, inputs):  # [None, rule_num]
        md = tf.expand_dims(tf.expand_dims(1.0 - inputs, -1), -1)  # [None, rule_num, 1, 1]
        mj = tf.expand_dims(tf.expand_dims(inputs, -1), -1)  # [None, rule_num, 1, 1]
        bc = tf.math.reduce_prod(mj * tf.nn.softmax(self.res) + md, -3) - tf.math.reduce_prod(md) + self.eps  # [None, res_dim, 2]
        return tf.math.log(bc[:, :, 1] / bc[:, :, 0])  # [None, res_dim]


class BRB(tf.keras.Model):
    def __init__(self, x, y):
        super(BRB, self).__init__()
        self.att = att_layer(x)
        self.dis = dis_layer(0.5 * tf.ones_like(x))
        self.res = res_layer(y)

    def custom_compile(self, att_trainable, dis_trainable, res_trainable):
        self.att.trainable = att_trainable
        self.dis.trainable = dis_trainable
        self.res.trainable = res_trainable
        self.compile(optimizer=tf.keras.optimizers.Adam(),  # SGD(momentum=0.99, nesterov=True)
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def call(self, inputs, training):
        out = self.res(self.dis(self.att(inputs)))
        if not training:
            out = tf.nn.softmax(out)
        return out


def create(x, y, rule_num, att_dim, res_dim, random_state, from_logits=False):
    if from_logits:
        y = np.argmax(y, -1)
    mask, rng = np.arange(x.shape[0]), np.random.default_rng(seed=random_state)
    rng.shuffle(mask)
    x, y = x[mask[:rule_num]], 2.0 * np.eye(res_dim)[y[mask[:rule_num]]] - 1.0
    y = np.concatenate((np.expand_dims(-y, -1), np.expand_dims(y, -1)), -1)
    return BRB(x, y)


def training(x, y, rule_num, att_dim, res_dim, epochs, batch_size, random_state):
    s = tf.distribute.MirroredStrategy()
    with s.scope():
        model = create(x, y, rule_num, att_dim, res_dim, random_state)
        model.custom_compile(True, True, True)

        tb = tf.keras.callbacks.TensorBoard(log_dir='logs', update_freq='batch')
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[])
    return model


def main():
    experiment_num, data_name = 20, "ecoli"
    rule_num, epoch, batch_size = 32, 4000, 128
    data, target, att_dim, res_dim = dataset_numeric_classification(data_name, 1)
    data = StandardScaler().fit_transform(data).astype(ndtype)
    sum_acc, sum_cnt = 0.0, 0
    for en in range(experiment_num):
        for train_data, train_target, test_data, test_target in kfold(data, target, 10, 'numeric', random_state=en):
            model = training(train_data, train_target,
                             rule_num, att_dim, res_dim,
                             epoch, batch_size, en)
            _, acc = model.evaluate(test_data, test_target)
            sum_acc += acc
            sum_cnt += 1
            tf.summary.scalar('acc_each_%s' % data_name, acc, sum_cnt)
            tf.summary.scalar('acc_%s' % data_name, sum_acc / sum_cnt, sum_cnt)


if __name__ == "__main__":
    main()
