import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold


class BRB:
    def __init__(self, junctive, has_metric, has_category, has_missing, rule_num, res_dim, **kwargs):
        self.RW = tf.Variable(np.random.normal(size=(rule_num,)), trainable=True, dtype=tf.float64)
        self.BC = tf.Variable(np.random.normal(size=(rule_num, res_dim,)), trainable=True, dtype=tf.float64)
        if has_metric:
            self.AM = tf.Variable(np.random.uniform(low=kwargs['low'], high=kwargs['high'], size=(rule_num, kwargs['metric_shape'])), trainable=True, dtype=tf.float64)
            self.OD = tf.Variable(np.log(kwargs['one']), trainable=True, dtype=tf.float64)
            self.WM = tf.math.square((self.AM - tf.expand_dims(kwargs['metric_input'], -2))/tf.math.exp(self.OD))
            # [rule_num,metric_shape] - [None,1,metric_shape] = [None,rule_num,metric_shape]
        if has_category:
            self.AC = [tf.Variable(np.random.normal(size=(rule_num, cs))) for cs in kwargs['category_shapes']]
            self.WC = tf.concat([tf.expand_dims(tf.distributions.kl_divergence(
                tf.distributions.Categorical(a), tf.distributions.Categorical(tf.expand_dims(ci, -2))), -1)
                for a, ci in zip(self.AC, kwargs['category_inputs'])], -1)
            # [[rule_num,cs] kl [None,1,cs]] = [[None,rule_num]] =ep,cat=> [None,rule_num,category_shapes[0]]
        if has_metric and has_category:
            self.W = tf.concat([self.WM, self.WC], -1)
        elif has_metric:
            self.W = self.WM
        elif has_category:
            self.W = self.WC
        else:
            raise Exception('no data type')
        # [None,rule_num,att_num]
        if has_missing:
            self.W = self.W * kwargs['missing_vector']
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


class Model_M_C:
    def __init__(self, metric_shape, low, high, one, category_shapes):
        self.XM = tf.placeholder(dtype=tf.float64, shape=[None, metric_shape])
        self.XC = [tf.placeholder(dtype=tf.float64, shape=[None, cs]) for cs in category_shapes]
        self.MV = tf.placeholder(dtype=tf.float64, shape=[None, metric_shape+len(category_shapes)])
        self.Y = tf.placeholder(dtype=tf.float64, shape=[None, 2])
        self.D = [BRB(junctive='dis', has_metric=True, has_category=True, has_missing=True, rule_num=32, res_dim=16,
                      metric_input=self.XM, metric_shape=metric_shape, low=low, high=high, one=one,
                      category_inputs=self.XC, category_shapes=category_shapes,
                      missing_vector=self.MV) for i in range(8)]
        self.E = BRB(junctive='con', has_metric=False, has_category=True, has_missing=False, rule_num=32, res_dim=2,
                     category_inputs=[d.Y for d in self.D], category_shapes=[16 for d in self.D])
        self.ERROR = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.Y, self.E.Y))
        self.STEP = tf.train.AdamOptimizer().minimize(self.ERROR)
        self.ACC = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.Y, -1), tf.argmax(self, self.E.Y, -1)), tf.float64))
        self.SESS = tf.Session()
        self.SESS.run(tf.global_variables_initializer())

    def train(self, metric_data, category_data, missing_vectors, target, ep=1000, bs=64):
        bn, fd, mask = int(np.ceil(len(target)/64)), {}, np.arange(len(target))
        for e in range(ep):
            np.random.shuffle(mask)
            md, cd, mv, t = metric_data[mask], [c[mask] for c in category_data], missing_vectors[mask], target[mask]
            for i in range(bn):
                fd[self.XM], fd[self.MV], fd[self.Y] = md[i*bs:(i+1)*bs], mv[i*bs:(i+1)*bs], t[i*bs:(i+1)*bs]
                for xc, c in zip(self.XC, cd):
                    fd[xc] = c[i*bs:(i+1)*bs]
                e, _ = self.SESS.run([self.ERROR, self.STEP], feed_dict=fd)
                print(e)

    def predict(self, metric_data, category_data, missing_vectors, target):
        fd = {}
        fd[self.XM], fd[self.MV], fd[self.Y] = metric_data, missing_vectors, target
        for xc, cd in zip(self.XC, category_data):
            fd[xc] = cd
        return self.SESS.run(self.ACC, feed_dict=fd)


def read_data():
    with open('../data/hepatitis.data') as f:
        for l in f:
            print(list(l.strip().split(',')))


def main():
    read_data()
    '''
    sample_num, metric_data, metric_shape, low, high, one, category_date, category_shapes, missing_vectors, target = read_data()
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
