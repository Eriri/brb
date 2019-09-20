
a = open('../data/iris.data', 'r')
b = open('../data/iris_rev.data', 'w')
c = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for i in a:
    e = i.strip().split(',')
    e[-1] = str(c.index(e[-1]))
    e = ' '.join(e)
    b.write(e+'\n')
    print(e)

b.close()
