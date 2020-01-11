import tensorflow as tf
import numpy as np
import os
from util import generate_variable, generate_junctive, evidential_reasoning
from util import replace_nan_with_zero, replace_zero_with_one, lookup, kl_divergence, bha_distance
from util import dataset_adult, kfold, training, evaluating

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class BaseN(tf.Module):
    def __init__(self, rule_num, att_dim, low, high, res_dim, junc, name=None):
        super(BaseN, self).__init__(name=name)
        self.a = generate_variable((rule_num, att_dim), initial_value=np.random.uniform(low, high, (rule_num, att_dim,)))
        self.o = generate_variable((att_dim,), initial_value=0.5 * (high - low))
        self.dz = tf.constant(0.01 * (high - low))
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))
        self.j = generate_junctive(junc)
        self.tv = self.trainable_variables

    @tf.Module.with_name_scope
    def __call__(self, x):  # (None, att_dim)
        w = tf.math.exp(replace_nan_with_zero(-tf.math.square(self.a - tf.expand_dims(x, -2)) / (tf.math.square(self.o)+self.dz)))
        return evidential_reasoning(self.j(w) * tf.math.exp(self.r), tf.nn.softmax(self.b))


class BaseC(tf.Module):
    def __init__(self, rule_num, cat_dims, res_dim, junc, name=None):
        super(BaseC, self).__init__(name=name)
        self.a = [generate_variable((rule_num, cat_dim)) for cat_dim in cat_dims]
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))
        self.j = generate_junctive(junc)
        self.tv = self.trainable_variables

    @tf.Module.with_name_scope
    def __call__(self, x):  # [(None), (None), ...]
        w = tf.concat(list(map(lambda x: lookup(x[0], x[1]), zip(x, self.a))), -1)
        return evidential_reasoning(self.j(w) * tf.math.exp(self.r), tf.nn.softmax(self.b))


class BaseM(tf.Module):
    def __init__(self, rule_num, att_dim, low, high, cat_dims, res_dim, junc, name=None):
        super(BaseM, self).__init__(name=name)
        self.an = generate_variable((rule_num, att_dim), initial_value=np.random.uniform(low, high, (rule_num, att_dim,)))
        self.o = generate_variable((att_dim,), initial_value=0.5 * (high - low))
        self.dz = tf.constant(0.01 * (high - low))
        self.ac = [generate_variable((rule_num, cat_dim,)) for cat_dim in cat_dims]
        self.b = generate_variable((rule_num, res_dim,))
        self.r = generate_variable((rule_num,))
        self.j = generate_junctive(junc)
        self.tv = self.trainable_variables

    @tf.Module.with_name_scope
    def __call__(self, x):  # (None, att_dim), [(None), (None), ...]
        w = tf.concat(list(map(lambda x: lookup(x[0], x[1]), zip(x[1], self.ac))) +
                      [tf.math.exp(replace_nan_with_zero(-tf.math.square(self.an - tf.expand_dims(x[0], -2)) / (tf.math.square(self.o)+self.dz)))], -1)
        return evidential_reasoning(self.j(w) * tf.math.exp(self.r), tf.nn.softmax(self.b))


class Dist(tf.Module):
    def __init__(self, rule_num, dis_num, dis_dim, res_dim, junc, name=None):
        super(Dist, self).__init__(name=name)
        self.a = generate_variable((rule_num, dis_num, dis_dim,))
        self.b = generate_variable((rule_num, res_dim))
        self.r = generate_variable((rule_num,))
        self.j = generate_junctive(junc)

    @tf.Module.with_name_scope
    def __call__(self, x):  # (None, dis_num, dis_dim)
        # w = kl_divergence(tf.nn.softmax(self.a), tf.expand_dims(x, -3))
        w = bha_distance(tf.nn.softmax(self.a), tf.expand_dims(x, -3))
        return evidential_reasoning(self.j(w)*tf.exp(self.r), tf.nn.softmax(self.b))


class Model(tf.Module):
    def __init__(self, att_dim, low, high, cat_dims, name=None):
        super(Model, self).__init__(name=name)
        self.base = [BaseM(rule_num=16, att_dim=att_dim, low=low, high=high,
                           cat_dims=cat_dims, res_dim=4, junc='dis', name=('base_%d' % i)) for i in range(4)]
        self.between = [Dist(rule_num=16, dis_num=4, dis_dim=4, res_dim=2, junc='dis', name=('between_%d' % i)) for i in range(2)]
        # self.final = Dist(rule_num=16, dis_num=4, dis_dim=5, res_dim=2, junc='dis', name=('final_%d' % i)) for i in range(4)]
        self.final = Dist(rule_num=64, dis_num=2, dis_dim=2, res_dim=2, junc='con', name='final')

        self.tv = self.trainable_variables

    @tf.Module.with_name_scope
    def __call__(self, x):
        x_base_out = tf.concat([tf.expand_dims(b(x), -2) for b in self.base], -2)
        x_between_out = tf.concat([tf.expand_dims(b(x_base_out), -2) for b in self.between], -2)
        x_final_out = self.final(x_between_out)
        # x_final_out = tf.reduce_mean(tf.concat([tf.expand_dims(b(x_between_out), -2) for b in self.final], -2), -2)
        return x_final_out


def main():
    x, cs, y = dataset_adult()
    for train_x, train_y, test_x, test_y in kfold(x, y, 10, 'mixed'):
        model = Model(train_x[0].shape[1], np.nanmin(train_x[0], 0), np.nanmax(train_x[0], 0), cs, 'brb')
        ebrb = BaseM(32, train_x[0].shape[1], np.nanmin(train_x[0], 0), np.nanmax(train_x[0], 0), cs, 2, 'con')
        training(ebrb, train_x, train_y)
        evaluating(ebrb, test_x, test_y, 'acc')
        # training(model, train_x, train_y)
        # evaluating(model, test_x, test_y, 'acc')
        exit(0)


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float32')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    main()
