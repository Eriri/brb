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


def result_category(y_true, y_pred):
    res = dict()
    res['acc'] = accuracy_score(y_true, y_pred)


def train(model, has_metric, has_category, target_type, fold=5, et=50):
    result = []
    for ex in range(et):
