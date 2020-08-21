import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

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


def dataset_numeric_classification(name, version='active'):
    ds = fetch_openml(name=name, version=version)
    data, target = ds['data'], ds['target']
    pot = list(set(target))
    target = np.array(np.vectorize(lambda x: pot.index(x))(target), np.int32)
    return data, target, data.shape[-1], len(pot)

def missing_at_random(data, rate, random_state=None):
    en = np.prod(date.shape)
    men = np.ceil(en * rate)
    mask = np.array(shuffle([False] * (en - men) + [True] * men, random_state=random_state))
    data[mask.reshape(data.shape)] = np.nan
    return data

def dataset_oil():
    x, y = [], []
    with open('../data/oil.data') as f:
        for i in f:
            z = list(map(float, i.split()))
            x.append(z[:2]), y.append(z[-1])
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
