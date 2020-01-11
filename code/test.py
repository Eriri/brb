import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold


# ds = fetch_openml('adult', 2)
# fn, cg = ds['feature_names'], ds['categories']
# nis = [fn.index(i) for i in fn if i not in cg.keys() and i != 'fnlwgt']
# cis = [fn.index(i) for i in fn if i in cg.keys()]
# cgs = [len(cg[att]) for att in cg.keys()]
# nume, cate = ds['data'][:, nis], ds['data'][:, cis]

# res, pot = ds['target'], list(set(ds['target']))
# res = np.vectorize(lambda x: pot.index(x))(res)

# data = (nume, tuple([c for c in np.transpose(cate)]))

# for train_mask, test_mask in KFold(n_splits=10).split(res):
#     data = tuple(map(lambda array: array[train_mask], data[1]))
#     print(data)
#     exit(0)


a = [tf.constant([0, 1, 2, 1]), tf.constant([0, 1])]
b = [tf.constant([[1, 2], [3, 4], [5, 6]]), tf.constant([[1, 2, 3], [4, 5, 6]])]

c = list(map(lambda x: tf.nn.embedding_lookup(x[1], x[0]), zip(a, b)))

print(c)
