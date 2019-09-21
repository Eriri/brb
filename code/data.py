
a = open('../data/haberman.data', 'r')
b = open('../data/haberman_rev.data', 'w')
c = ['1', '2']

for i in a:
    e = i.strip().split(',')
    e[-1] = str(c.index(e[-1]))
    e = ' '.join(e)
    b.write(e+'\n')
    print(e)

b.close()
