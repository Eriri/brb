import pandas as pd

d = pd.merge(pd.read_csv('../data/bank/train.csv'), pd.read_csv('../data/bank/train_target.csv'))
print(set(d['ncloseCreditCard']))

d.drop(['id', 'certId', 'dist', 'certValidBegin', 'certValidStop'], axis=1, inplace=True)
d.drop(['bankCard', 'isNew', 'residentAddr', 'setupHour', 'weekday'], axis=1, inplace=True)
d.drop(['x_%d' % d for d in range(79)], axis=1, inplace=True)
d.drop(['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan'], axis=1, inplace=True)


d.to_csv('train_rev.csv')
