import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, StratifiedKFold


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


def kfold(x, y, n_splits, stratified=False):
    kf = KFold(n_splits=n_splits) if not stratified else StratifiedKFold(n_splits=n_splits)
    data = []
    for train_mask, test_mask in kf.split(x, y):
        data.append((x[train_mask], y[train_mask], x[test_mask], y[test_mask]))
    return data


def generate_variable_numerical(rule_num, att_dim, low, high, dtype=tf.float64, trainable=True):
    '''[rule_num,att_dim]'''
    return tf.Variable(np.random.uniform(low, high, size=(rule_num, att_dim,)), trainable=trainable, dtype=dtype)


def generate_variable_categorical(rule_num, att_dims, dtype=tf.float64, trainable=True):
    '''[[rule_num,att_dim],...]'''
    return [tf.Variable(np.random.normal(size=(rule_num, att_dim)), trainable=trainable, dtype=dtype) for att_dim in att_dims]


def genertae_normal(shape, dtype=tf.float64, trainable=True):
    return tf.Variable(tf.random.normal(shape=shape, dtype=dtype), trainable=trainable, dtype=dtype)


def replace_nan_with_zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def replace_zero_with_one(x):
    return tf.where(tf.math.equal(x, tf.zeros_like(x)), tf.ones_like(x), x)


def activating_weight_categorical(x, a, rw, junc):
    '''
    [[None,att_dim],...],[[rule_num,att_dim],...],[rule_num]->[None,rule_num]
    a normalized, rw not negative
    '''
    w = [tf.reduce_sum(tf.math.sqrt(ai * tf.expand_dims(xi, -2)), -1) for xi, ai in zip(x, a)]
    if junc == 'con':
        return tf.reduce_prod(tf.concat([tf.expand_dims(replace_zero_with_one(wi), -1) for wi in w], -1), -1)
    if junc == 'dis':
        return tf.reduce_sum(tf.concat([tf.expand_dims(wi, -1) for wi in w], -1), -1)
    raise Exception('junc should be either con or dis')


def activating_weight_numerical(x, a, o, rw, junc):
    '''
    [None,att_dim],[rule_num,att_dim],[att_dim],[rule_num]->[None,rule_num]
    o and rw not negative
    '''
    w = tf.math.square((a - tf.expand_dims(x, -2))/o)
    if junc == 'con':
        return tf.math.exp(-tf.reduce_sum(replace_nan_with_zero(w), -1)) * rw
    if junc == 'dis':
        return tf.reduce_sum(replace_nan_with_zero(tf.math.exp(tf.negative(w))), -1) * rw
    raise Exception('junc should be either con or dis')


def evidential_reasoning(aw, beta):  # [None,rule_num],[rule_num,res_dim]->[None,res_dim]
    bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * beta + 1.0, -2) - 1.0
    return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)
