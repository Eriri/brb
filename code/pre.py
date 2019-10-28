import pandas as pd
import numpy as np

d = pd.merge(pd.read_csv('../data/bank/train.csv'), pd.read_csv('../data/bank/train_target.csv'))
print(set(d['ncloseCreditCard']))

d.drop(['id', 'certId', 'dist', 'certValidBegin', 'certValidStop'], axis=1, inplace=True)
d.drop(['bankCard', 'isNew', 'residentAddr', 'setupHour', 'weekday'], axis=1, inplace=True)
d.drop(['x_%d' % d for d in range(79)], axis=1, inplace=True)
d.drop(['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan'], axis=1, inplace=True)


d.to_csv('../data/bank/train_rev.csv')

df = pd.read_csv('train_rev.csv')
matt = ['age', 'lmt']
catt = ['loanProduct', 'gender', 'edu', 'job', 'basicLevel', 'ethnic', 'highestEdu', 'linkRela']
mdata = np.transpose(np.array([np.array(df[m]) for m in matt]))
one, low, high = np.ptp(mdata, axis=0), np.min(mdata, axis=0), np.max(mdata, axis=0)
ls, ts = [list(set(df[c])) for c in catt], list(set(df['target']))
cdata = [np.zeros((df.shape[0], len(l),)) for l in ls]
tdata = np.zeros((df.shape[0], len(ts),))
for i, r in df.iterrows():
    for di, ci, li in zip(cdata, catt, ls):
        di[i][li.index(r[ci])] = 1.0
    tdata[i][ts.index(r['target'])] = 1.0
for a, c in zip(catt, cdata):
    np.savez('../data/bank/np/train_cdata_%s' % a, c)
np.savez('../data/bank/np/train_mdata', mdata)
np.savez('../data/bank/np/train_tdata', tdata)
