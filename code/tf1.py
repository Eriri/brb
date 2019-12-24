import tensorflow as tf
import numpy as numpy
from util import *


class Base:
    def __init__(self, data_type, rule_num, res_dim, junctive, **kwargs):
        self.data_type, self.junctive = data_type, junctive
        if self.data_type == 'numerical' or self.data_type == 'mixed':
            att_dim, low, high, one = kwargs['att_dim'], kwargs['low'], kwargs['high'], kwargs['one']
            self.an = generate_variable(shape=(rule_num, att_dim,), initial_value=np.random.uniform(low, high, size=(rule_num, att_dim,)))
            self.on = generate_variable(shape=(att_dim,), initial_value=np.sqrt(one))
        if self.data_type == 'categorical' or self.data_type == 'mixed':
            self.cat_dims = kwargs['cat_dims']
            self.ac = [generate_variable(shape=(rule_num, cat_dim)) for cat_dim in self.cat_dims]
            self.oc = generate_variable(shape=(len(self.cat_dims),), initial_value=np.ones(shape=(len(self.cat_dims))))
        if self.data_type == 'distribution':
            dis_num, dis_dim = kwargs['dis_num'], kwargs['dis_dim']
            self.ad = generate_variable(shape=(rule_num, dis_num, dis_dim,))
        self.r = generate_variable(shape=(rule_num,))
        self.b = generate_variable(shape=(rule_num, res_dim,))

    def __call__(self, x):

        if self.data_type == 'numerical' or self.data_type == 'mixed':
            activating_weight_numerical(x, self.an, tf.math.square(self.on) + tf.constant(1e-9), tf.nn.si
            pass
        if self.data_type == 'categorical' or self.data_type == 'mixed':
            pass
        if self.data_type == 'distribution':
            pass


class Model:
    def __init__(self):

        pass

    def train(self, x, y, bs, ep):
        pass

    def evaluate(self, x, y):
        pass

    def predict(self, x):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
