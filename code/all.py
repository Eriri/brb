import tensorflow as tf
import numpy as np
from util import *


def hellinger_distance(ac, ci):
    '''[rule_num,cs] [None,1,cs] [None,rule_num]'''
    return tf.math.sqrt(1.0-tf.reduce_sum(tf.math.sqrt(tf.nn.softmax(ac)*tf.expand_dims(ci, -2)), -1))


def bhattacharyya_distance(ac, ci):
    '''[rule_num,cs] [None,1,cs] [None,rule_num]'''
    return tf.negative(tf.log(tf.reduce_sum(tf.math.sqrt(tf.nn.softmax(ac), tf.expand_dims(ci, -2)), -1)))


class BRB:
    def __init__(self, junctive, has_metric, has_category, rule_num, res_dim, **kwargs):
        self.RW = tf.Variable(np.random.normal(size=(rule_num,)), trainable=True, dtype=tf.float64)
        self.BC = tf.Variable(np.random.normal(size=(rule_num, res_dim,)), trainable=True, dtype=tf.float64)
        self.WT = []
        if has_metric:
            self.AM = tf.Variable(np.random.uniform(low=kwargs['low'], high=kwargs['high'], size=(rule_num, kwargs['metric_shape'])), trainable=True, dtype=tf.float64)
            self.OD = tf.Variable(np.log(kwargs['one']), trainable=True, dtype=tf.float64)
            self.WT.append(tf.math.square((self.AM - tf.expand_dims(kwargs['metric_input'], -2))/tf.math.exp(self.OD)))
            # [rule_num,metric_shape] - [None,1,metric_shape] = [None,rule_num,metric_shape]
        if has_category:
            self.AC = [tf.Variable(np.random.normal(size=(rule_num, cs))) for cs in kwargs['category_shapes']]
            self.WT.append(tf.concat([tf.expand_dims(hellinger_distance(ac, ci), -1) for ac, ci in zip(self.AC, kwargs['category_inputs'])], -1))
            # [rule_num,cs] [None,1,cs] [[None,rule_num,-1]] [None,rule_num,category_shapes[0]]
        self.WF = tf.concat(self.WT, -1)
        self.W = tf.where(tf.is_nan(self.WF), tf.ones_like(self.WF), self.WF)  # [None,rule_num,att_num]
        if junctive == 'con':
            self.AW = tf.math.exp(-tf.reduce_sum(self.W, -1)) * tf.exp(self.RW)
        elif junctive == 'dis':
            self.AW = tf.reduce_sum(1.0 / tf.math.exp(self.W), -1) * tf.exp(self.RW)
        else:
            raise Exception('parameter junctive is either dis or con')
        # [None,rule_num]
        self.SW = tf.reduce_sum(self.AW, -1)
        self.B = tf.reduce_prod(tf.expand_dims(self.AW/(tf.expand_dims(self.SW, -1)-self.AW), -1)*tf.nn.softmax(self.BC)+1.0, -2)-1.0
        self.Y = self.B / tf.expand_dims(tf.reduce_sum(self.B, -1), -1)
        # [None,res_dim]


class BaseModel:
    def __init__(self, metric_shape, low, high, one, category_shapes):
        self.XM = tf.placeholder(dtype=tf.float64, shape=[None, metric_shape])
        self.XC = tf.placeholder(dtype=tf.int64, shape=[None, len(category_shapes)])
        self.XCO = [tf.one_hot(xc, cs) for xc, cs in zip(self.XC, category_shapes)]
        self.Y = tf.placeholder(dtype=tf.float64, shape=[None, 2])
        self.D = [BRB(junctive='dis', has_metric=True, has_category=True, has_missing=True, rule_num=32, res_dim=16,
                      metric_input=self.XM, metric_shape=metric_shape, low=low, high=high, one=one,
                      category_inputs=self.XCO, category_shapes=category_shapes) for i in range(8)]
        self.E = BRB(junctive='con', has_metric=False, has_category=True, has_missing=False, rule_num=32, res_dim=2,
                     category_inputs=[tf.math.log(d.Y) for d in self.D], category_shapes=[16 for d in self.D])
        self.ERROR = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.Y, self.E.Y))
        self.STEP = tf.train.AdamOptimizer().minimize(self.ERROR)
        self.SESS = tf.Session()
        self.SESS.run(tf.global_variables_initializer())

    def train(self, metric_data, category_data, target, ep=10, bs=64):
        bn, mask = int(np.ceil(len(target)/bs)), np.arange(len(target))
        for e in range(ep):
            np.random.shuffle(mask)
            md, cd, t = metric_data[mask], [c[mask] for c in category_data], target[mask]
            for i in range(bn):
                fd = {self.XM: md[i*bs:(i+1)*bs], self.XC: cd[i*bs:(i+1)*bs], self.Y: t[i*bs:(i+1)*bs]}
                e, _ = self.SESS.run([self.ERROR, self.STEP], feed_dict=fd)
                print(i, e)

    def predict(self, metric_data, category_data):
        bn, res = int(np.ceil(metric_data.shape[0]/64)), []
        for i in range(bn):
            fd = {self.XM: metric_data[i*64:(i+1)*64], self.XC: category_data[i*64:(i+1)*64]}
            res.append(self.SESS.run(self.E.Y, feed_dict=fd))
        return np.concatenate(res)


def main():
    '''
    sample_num, metric_data, metric_shape, low, high, one, category_date, category_shapes, missing_vector, target = ()
    mask, ep, kf = np.arange(sample_num), 50, KFold(n_splits=5)
    for e in range(ep):
        np.random.shuffle(mask)
        for train_mask, test_mask in kf.split(mask):
            train_md, test_md = metric_data[mask[train_mask]], metric_data[mask[test_mask]]
            train_cd, test_cd = category_date[mask[train_mask]], category_date[mask[test_mask]]
            train_mv, test_mv = missing_vectors[mask[train_mask]], missing_vectors[mask[test_mask]]
            train_t, test_t = target[mask[train_mask]], target[mask[test_mask]]
            z = Model_M_C(metric_shape, low, high, one, category_shapes)
            z.train(train_md, train_cd, train_mv, train_t)
            print(z.predict(test_md, test_cd, test_mv, test_t))
    '''


if __name__ == "__main__":
    main()
