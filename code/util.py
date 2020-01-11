import numpy as np
import tensorflow as tf
import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, StratifiedKFold


def dataset_adult():
    ds = fetch_openml('adult', 2)
    fn, cg = ds['feature_names'], ds['categories']
    nis = [fn.index(i) for i in fn if i not in cg.keys() and i != 'fnlwgt']
    cis = [fn.index(i) for i in fn if i in cg.keys()]
    cgs = [len(cg[att]) for att in cg.keys()]
    res, pot = ds['target'], list(set(ds['target']))
    res = np.array(np.vectorize(lambda x: pot.index(x))(res), np.int32)
    nume, cate = np.array(ds['data'][:, nis], np.float32), np.array(ds['data'][:, cis], np.float32)
    cate = tuple([c for c in np.transpose(cate)])
    return (nume, cate), cgs, res


def read_oil():
    x, y = [], []
    with open('../data/oil.data') as f:
        for i in f:
            z = list(map(float, i.split()))
            x.append(z[:2]), y.append(z[-1])
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def kfold(x, y, n_splits, dtype, stratified=False, shuffle=True, random_state=None):
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if dtype == 'numeric':
        return [(x[train_mask], y[train_mask], x[test_mask], y[test_mask])
                for train_mask, test_mask in kf.split(np.arange(y.shape[0]), y)]
    if dtype == 'categorical':
        return [(tuple(map(lambda x: x[train_mask], x)), y[train_mask],
                 tuple(map(lambda x: x[test_mask], x), y[test_mask]))
                for train_mask, test_mask in kf.split(np.arange(y.shape[0]), y)]
    if dtype == 'mixed':
        return [((x[0][train_mask], tuple(map(lambda x: x[train_mask], x[1]))), y[train_mask],
                 (x[0][test_mask], tuple(map(lambda x: x[test_mask], x[1]))), y[test_mask])
                for train_mask, test_mask in kf.split(np.arange(y.shape[0]), y)]
    raise Exception('dtype error')


def random_replace_with_nan(x, p):
    return np.where(np.random.binomial(1, p, x.shape).astype(np.bool), np.full(x.shape, np.nan), x)


def missing(x, p, dtype):
    if dtype == 'numeric':
        return random_replace_with_nan(x, p)
    if dtype == 'categorical':
        return tuple(map(lambda x: random_replace_with_nan(x, p), x))
    if dtype == 'mixed':
        return (random_replace_with_nan(x[0], p), tuple(map(lambda x: random_replace_with_nan(x, p), x[1])))
    raise Exception('dtype error')


def training(model, x, y, ep=10, bs=64):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    opt = tf.optimizers.Adam()
    total = ep * np.ceil(y.shape[0] / bs)
    for cnt, (xi, yi) in enumerate(ds):
        with tf.GradientTape(persistent=True) as gt:
            pi = model(xi)
            acc = tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(yi, pi))
            loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(yi, pi))
        opt.apply_gradients(zip(gt.gradient(loss, model.tv), model.tv))
        print("loss=%.6f, acc=%.6f, %.3lf%" % (loss.numpy(), acc.numpy(), cnt/total), end='\r')


def evaluating(model, x, y, mtype):
    if mtype == 'acc':
        print(tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(tf.constant(y), model(x))))
        return
    if mtype == 'mae':
        print(tf.metrics.mean_absolute_error(tf.constant(y), model(x)))
        return


def generate_variable(shape, dtype=tf.float32, trainable=True, initial_value=None):
    if initial_value is None:
        return tf.Variable(tf.random.normal(shape=shape, dtype=dtype), trainable=trainable, dtype=dtype)
    return tf.Variable(initial_value=initial_value, shape=shape, trainable=trainable, dtype=dtype)


def generate_junctive(junc):
    if junc == 'con':
        return tf.function(lambda x: tf.reduce_prod(x, -1))
    if junc == 'dis':
        return tf.function(lambda x: tf.reduce_sum(x, -1))
    raise Exception('junctive error')


@tf.function
def replace_nan_with_zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


@tf.function
def replace_zero_with_one(x):
    return tf.where(tf.math.equal(x, tf.zeros_like(x)), tf.ones_like(x), x)


@tf.function
def evidential_reasoning(aw, beta):  # [None,rule_num],[rule_num,res_dim]->[None,res_dim]
    bc = tf.reduce_prod(tf.expand_dims(aw / (tf.expand_dims(tf.reduce_sum(aw, -1), -1) - aw), -1) * beta + 1.0, -2) - 1.0
    return bc / tf.expand_dims(tf.reduce_sum(bc, -1), -1)


@tf.function
def lookup(x, a):  # [None],[rule_num, cat_dim]->[None,rule_num,1]
    return tf.expand_dims(replace_zero_with_one(tf.nn.embedding_lookup(
        tf.transpose(tf.nn.softmax(a)),
        tf.cast(tf.where(tf.math.is_nan(x), -tf.ones_like(x), x), tf.int32))), -1)


@tf.function
def kl_divergence(x, y):
    return tf.reduce_sum(x * (tf.math.log(x) - tf.math.log(y)), -1)


@tf.function
def js_divergence(x, y):
    return kl_divergence(x, 0.5 * (x + y)) + kl_divergence(y, 0.5 * (x + y))


@tf.function
def bha_distance(x, y):
    return -tf.math.log(tf.reduce_sum(tf.math.sqrt(x * y), -1))
