import numpy as np
from sklearn.datasets import fetch_openml


def read_data_m_c_c(dataset_name):
    ds = fetch_openml(name=dataset_name)
    sn = ds['target'].shape[0]
    fn, cg, tn = list(ds['feature_names']), dict(ds['categories']), list(set(ds['target']))
    mk, ck, ti = set(fn) - set(cg.keys()), set(cg.keys()), [tn.index(t) for t in ds['target']]
    mi, ci, cl = [fn.index(k) for k in mk], [fn.index(k) for k in ck], [len(cg[k]) for k in ck]
    metric_data, category_raw = ds['data'][:, mi], ds['data'][:, ci]
    low, high = np.nanmin(metric_data, axis=0), np.nanmax(metric_data, axis=0)
    one, metric_missing = high - low, []
    for i in range(metric_data.shape[0]):
        mmv = np.ones((metric_data.shape[1],))
        for j in range(metric_data.shape[1]):
            if metric_data[i, j] is np.nan:
                metric_data[i, j], mmv[j] = np.random.uniform(low[j], high[j]), 0.0
        metric_missing.append(mmv)
    category_data, category_missing = [[]] * category_raw.shape[1], []
    for i in range(category_raw.shape[0]):
        mmv = np.ones((category_raw.shape[1],))
        for j in range(category_raw.shape[1]):
            dis, k = -np.ones((cl[j],)), category_raw[i, j]
            if k is not np.nan:
                dis[np.int(k)] = 1.0
            else:
                mmv[j] = 0.0
            category_data[j].append(dis)
        category_missing.append(mmv)
    target = [np.zeros((len(tn,))) for i in range(sn)]
    for i in range(sn):
        target[i][ti[i]] = 1.0
    category_data = [np.array(cd) for cd in category_data]
    missing_vector = np.concatenate((np.array(metric_missing), np.array(category_missing)), -1)
    return sn, metric_data, metric_data.shape, low, high, one, category_data, [c.shape for c in category_data], missing_vector, target
