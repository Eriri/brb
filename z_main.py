import math
import functools
import operator


class Rule:
    def __init__(self, a, c, att, res):
        self.ant = trans(a, att)
        self.con = [0.0]*len(res)
        self.con[res.index(c)] = 1.0
        self.rwei, self.awei, self.uwei = 0.0, 0.0, 0.0


def trans(a, att):  # raw data to belief structure
    t = [[0.0]*len(x) for x in att]
    for i in range(len(att)):
        if a[i] in att[i]:
            t[i][att[i].index(a[i])] = 1
        else:
            for j in range(len(att[i])-1):
                if att[i][j] < a[i] and a[i] < att[i][j+1]:
                    t[i][j] = (att[i][j+1] - a[i]) / (att[i][j+1] - att[i][j])
                    t[i][j+1] = (a[i] - att[i][j]) / (att[i][j+1] - att[i][j])
                    break
    return t


def simi(a, b):  # calculate similarity of two belief distribution
    assert len(a) == len(b)
    return max(0.0, 1.0-math.sqrt(sum([(c-d)*(c-d) for c, d in zip(a, b)])))


def arw(rules):  # assign rule weight
    incon = [1.0*(len(rules)-1)] * len(rules)
    for i in range(len(rules)):
        for j in range(len(rules)):
            if i != j:
                sra = min([simi(rules[i].ant[k], rules[j].ant[k])
                           for k in range(len(attributes))])
                src = simi(rules[i].con, rules[j].con)
            if sra != 0.0 and src != 0.0:
                incon[i] -= math.exp(-sra*sra*(sra/src-1.0)*(sra/src-1.0))
    sigincon = sum(incon)
    for i in range(len(rules)):
        rules[i].rwei = 1.0 if sigincon == 0.0 else 1.0 - incon[i]/sigincon


def caw(r, rules):  # calculate activate weight
    sw = 0.0
    for rule in rules:
        rule.awei = functools.reduce(operator.mul, [simi(
            r.ant[i], rule.ant[i]) for i in range(len(attributes))], rule.rwei)
        sw += rule.awei
    for rule in rules:
        rule.awei /= sw


def er(rules):  # evidential reasoning
    for rule in rules:


if __name__ == "__main__":
    f = open('glass/Testdata_214.txt')
    data = [x.split() for x in f.readlines()]
    data = [list(map(float, x)) for x in data]
    global attributes, results
    attributes = [[min(x)+y*(max(x)-min(x))/4 for y in range(5)]
                  for x in zip(*data)][:-1]
    results = [0, 1, 2, 3, 4, 5, 6]
    rules = [Rule(x[:-1], x[-1], attributes, results) for x in data]
