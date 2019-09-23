
f = open('../data/oil_rev.txt')
c = []
for l in f:
    c.append(float(l.strip().split()[-1]))

print(min(c), max(c))

# a = open('../data/glass.data', 'r')
# b = open('../data/glass_rev.data', 'w')
# c = ['1', '2', '3', '5', '6', '7']

# for i in a:
#     e = i.strip().split(',')
#     e[-1] = str(c.index(e[-1]))
#     e = ' '.join(e)
#     b.write(e+'\n')
#     print(e)

# b.close()
