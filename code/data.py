
a = open('../data/glass.data', 'r')
b = open('../data/glass_rev.data', 'w')
c = ['1', '2', '3', '5', '6', '7']

for i in a:
    e = i.strip().split(',')
    e[-1] = str(c.index(e[-1]))
    e = ' '.join(e)
    b.write(e+'\n')
    print(e)

b.close()
