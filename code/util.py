import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score


def read_data(dataset_name):
    ds = fetch_openml(name=dataset_name)
    sn = ds['target'].shape[0]
    fn, cg, tn = list(ds['feature_names']), ds['categories'], np.array(list(set(ds['target'])))
    mkeys, ckeys = list(set(fn) - set(cg.keys())), list(set(cg.keys()))
    mindex, cindex = [fn.index(k) for k in mkeys], [fn.index(k) for k in ckeys]
    cshapes = [len(cg[k]) for k in ckeys]
    md, cd = ds['data'][:, mindex], ds['data'][:, cindex]
    low, high = np.nanmin(md, axis=0), np.nanmax(md, axis=0)
    target = np.array(np.expand_dims(cd, -1) == tn, np.float)
    return sn, md, md.shape[1], low, high, high - low, cd, cshapes, target


def read_oil():
    x, y = [], []
    with open('../data/oil.data') as f:
        for i in f:
            z = list(map(float, i.split()))
            x.append(z[:2]), y.append(z[-1])
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def result_category(y_true, y_pred):
    res = dict()
    res['acc'] = accuracy_score(y_true, y_pred)


class BaseBRB:
    def __init__(self, rule_num, res_dim, junctive=None, has_cont=False, has_disc=False, **kwargs):
        '''
        junctive should be either con or dis
        if has_cont==True, please provide [low, high, one]
        if has_disc==True, please provide [disc_shapes]
        '''
        if junctive is None or junctive not in ['con', 'dis']:
            raise Exception('junctive should be either con or dis')
        self.junctive = junctive
        if has_cont:
            self.Ac = tf.Variable(np.random.uniform(kwargs['low'], kwargs['high'], size=(rule_num, kwargs['one'].shape[0],)), dtype=tf.float64, trainable=True)
            self.D = tf.Variable(np.log(kwargs['one']), dtype=tf.float64, trainable=True)
        if has_disc:
            self.Ad = [tf.Variable(np.random.normal(size=(rule_num, ds,)), dtype=tf.float64, trainable=True) for ds in kwargs['disc_shapes']]
        self.B = tf.Variable(tf.random.normal(shape=(rule_num, res_dim,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.R = tf.Variable(tf.zeros(shape=(rule_num,), dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.U = tf.Variable(kwargs['util'], dtype=tf.float64, trainable=True)

    def RIMER(self, inputs):
        '''
        inputs is a dict like {'cont':cont_data,'disc':disc_data}
        '''
        wt = []
        if 'cont' in inputs.keys():
            wt.append(tf.math.square((self.Ac - tf.expand_dims(inputs['cont'], -2)) / tf.math.exp(self.D)))
        if 'disc' in inputs.keys():
            for di, ad in zip(inputs['disc'], self.Ad):
                wt.append(tf.expand_dims(tf.math.sqrt(1.0-tf.reduce_sum(tf.math.sqrt(tf.nn.softmax(ad)*tf.expand_dims(di, -2)), -1)), -1))
        w = tf.concat(wt, -1)
        if self.junctive == 'con':
            aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.R)
        if self.junctive == 'dis':
            aw = tf.reduce_sum(tf.math.exp(tf.negative(w)), -1) * tf.math.exp(self.R)
        sw = tf.reduce_sum(aw, -1)
        bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(self.B)+1.0, -2)-1.0
        return tf.reduce_sum(bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1) * self.U, -1)


def RIMER_C_C(X, BRB):
    A, D, B, R = BRB
    w = tf.math.square((A - tf.expand_dims(X, -2)) / tf.math.exp(D))
    aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(R)
    sw = tf.reduce_sum(aw, -1)
    bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(B)+1.0, -2)-1.0
    return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)


def RIMER_C_D(X, BRB):
    A, D, B, R = BRB
    w = tf.math.square((A - tf.expand_dims(X, -2)) / tf.math.exp(D))
    aw = tf.reduce_sum(tf.math.exp(tf.negative(w)), -1) * tf.math.exp(D)
    sw = tf.reduce_sum(aw, -1)
    bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(B)+1.0, -2)-1.0
    return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)


def RIMER_D_C(X, BRB):
    A, D, B, R = BRB
    w = tf.concat([], -1)
    aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(R)
    sw = tf.reduce_sum(aw, -1)
    bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(B)+1.0, -2)-1.0
    return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)


def RIMER(X, BRB, junctive, datatype):
    XN, XC = X
    AN, AC, D, B, R = BRB
    w = []
    if datatype == 'both' or datatype == 'nume':
        w.append(tf.math.square((AN - tf.expand_dims(XN, -2)) / (tf.math.square(D)+0.1)))
    if datatype == 'both' or datatype == 'cate':
        for x, a in zip(XC, AC):
            w.append(tf.expand_dims(tf.math.sqrt(1.0-tf.reduce_sum(tf.math.sqrt(tf.nn.softmax(a)*tf.expand_dims(x, -2)), -1)), -1))
    w = tf.concat(w, -1)
    w = tf.where(tf.math.is_nan(w), tf.ones_like(w), w)
    if junctive not in ['con', 'dis']:
        raise Exception('junctive should be either con or dis')
    elif junctive == 'con':
        aw = tf.math.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(R)
    elif junctive == 'dis':
        aw = tf.reduce_sum(tf.math.exp(tf.negative(w)), -1) * tf.math.exp(R)
    sw = tf.reduce_sum(aw, -1)
    bc = tf.reduce_prod(tf.expand_dims(aw/(tf.expand_dims(sw, -1)-aw), -1)*tf.nn.softmax(B)+1.0, -2)-1.0
    return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
