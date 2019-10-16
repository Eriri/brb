import pandas as pd

d = pd.read_csv('../data/bank/train.csv')

x = sorted(list(set([(b-a) for a, b in zip(d['certValidBegin'], d['certValidStop'])])))
y = x[len(x)-1]
for i in range(len(x)):
    x[i] = int(x[i]/1e7)
x = set(x)
print(x)
print(len(x))
