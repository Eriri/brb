import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle


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


def missing_at_random(data, rate, random_state=None):
    en = np.prod(data.shape).astype(np.int)
    men = np.ceil(en * rate).astype(np.int)
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


def dataset_mammographic():
    x, y = [], []
    with open('../data/mammographic.data') as f:
        for i in f:
            z = list(i.split(','))
            for j in range(len(z)):
                if z[j] == '?':
                    z[j] = np.nan
            x.append(z[1:5]), y.append(z[-1])
    x, y = np.array(x).astype(np.float), np.array(y).astype(np.int)
    return x, y, x.shape[1], 2
