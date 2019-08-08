import math
import functools
import operator
import random

attributes, results, positive = [], [], []


class Rule:
    def __init__(self, a, c, rw=1.0, uw=0.0):
        self.ant = trans(a)
        self.con = [0.0]*len(results)
        self.con[results.index(c)] = 1.0
        self.rw, self.aw, self.uw = 0.0, rw, uw


def trans(a):  # raw data transform into belief structure
    t = [[0.0]*len(x) for x in attributes]
    for i in range(len(attributes)):
        if a[i] in attributes[i]:
            t[i][attributes[i].index(a[i])] = 1
        else:
            for j in range(len(attributes[i])-1):
                if attributes[i][j] < a[i] and a[i] < attributes[i][j+1]:
                    t[i][j] = (attributes[i][j+1] - a[i]) / \
                        (attributes[i][j+1] - attributes[i][j])
                    t[i][j+1] = (a[i] - attributes[i][j]) / \
                        (attributes[i][j+1] - attributes[i][j])
                    break
    return t


def simi(a, b):  # calculate similarity of two belief distribution
    assert len(a) == len(b)
    return max(0.0, 1.0-math.sqrt(sum([(c-d)*(c-d) for c, d in zip(a, b)])))


'''
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
        rules[i].rw = 1.0 if sigincon == 0.0 else 1.0 - incon[i]/sigincon
'''


def caw(r, rules):  # calculate activate weight
    sw = 0.0
    for rule in rules:
        rule.aw = functools.reduce(operator.mul, [simi(
            r.ant[i], rule.ant[i]) for i in range(len(attributes))], rule.rw)
        sw += rule.aw
    if sw == 0.0:
        return
    for rule in rules:
        rule.aw /= sw


def er(rules):  # evidential reasoning
    B = [0.0] * len(results)
    D = functools.reduce(operator.mul, [1.0-r.aw for r in rules], 1.0)
    for k in range(len(results)):
        B[k] = functools.reduce(
            operator.mul, [r.aw*r.con[k]+1.0-r.aw for r in rules], 1.0)
    S = sum(B)
    if S == len(results)*D:
        for k in range(len(results)):
            B[k] = 1.0/len(results)
        return B
    for k in range(len(results)):
        B[k] = (B[k]-D)/(S-len(results)*D)
    return B


def cs(rules, data):  # calculate score
    error = 0.0
    for r in data:
        caw(r, rules)
        B = er(rules)
        error += r.uw * (B.index(max(B)) != r.con.index(max(r.con)))
    return error


def de(rules, data):
    pwei, pe = [random.random() for i in range(len(rules))], float("inf")
    bwei, be = pwei, float("inf")
    cnt = 0
    while cnt < 10:
        nwei = pwei
        for i in range(len(rules)):
            if random.random() < 0.5:
                nwei[i] = random.random()
            if random.random() < 0.5:
                nwei[i] = pwei[i]
        for i in range(len(rules)):
            rules[i].rw = nwei[i]
        ne = cs(rules, data)
        if ne < pe:
            cnt = 0
            pwei, pe = nwei, ne
            if ne < be:
                bwei, be = nwei, ne
        else:
            cnt += 1
    for i in range(len(rules)):
        rules[i].rw = bwei[i]


def adacost(train, test, valid):
    base, weight = [], []
    for i in range(len(test)):
        test[i].uw = 1.0 / len(test)
    for i in range(10):
        b = []
        for i in range(len(test)):
            b.append(train[random.randint(0, len(train)-1)])
        de(b, test)
        a = 0.5 * math.log(1.0/cs(b, test)-1) + math.log(len(results)-1)
        base.append(b)
        weight.append(a)
        for r in test:
            caw(r, b)
            B = er(b)
            tar, pre = r.con.index(max(r.con)), B.index(max(B))
            r.uw *= math.exp((1-2*(tar == pre)+0.5*(tar in positive))*a)
        s = sum([r.uw for r in test])
        for r in test:
            r.uw /= s
    A = [[0] * len(results)] * len(results)
    for r in valid:
        T = [0.0] * len(results)
        for b, w in zip(base, weight):
            caw(r, b)
            B = er(b)
            for i in range(len(results)):
                T[i] += w*B[i]
        A[r.con.index(max(r.con))][T.index(max(T))] += 1
    accuracy = sum([A[i][i] for i in range(len(results))])/sum(map(sum, A))
    precision = A[5][5]/(sum([A[i][5] for i in range(len(results))])+1)
    recall = A[5][5]/(sum([A[5][i] for i in range(len(results))])+1)
    f1 = precision*recall/(precision+recall+1)
    return accuracy, precision, recall, f1


def kf(rules, k):  # 10 k fold
    d = int(len(rules)/10)
    valid = rules[d*k:d*(k+1)]
    remain = rules[:d*k]+rules[d*(k+1):]
    random.shuffle(remain)
    return remain[int(len(remain)/4):], remain[:int(len(remain)/4)], valid


if __name__ == "__main__":
    f = open('Testdata_214.txt')
    data = [x.split() for x in f.readlines()]
    data = [list(map(float, x)) for x in data]
    attributes = [[min(x)+y*(max(x)-min(x))/4 for y in range(5)]
                  for x in zip(*data)][:-1]
    results = [0, 1, 2, 3, 4, 5, 6]
    positive = [5]
    rules = [Rule(x[:-1], x[-1]) for x in data]
    random.shuffle(rules)
    cnt = 0
    for r in rules:
        caw(r, rules)
        B = er(rules)
        cnt += B.index(max(B)) == r.con.index(max(r.con))
    print(cnt)

    sum_acc, sum_pre, sum_rec, sum_f1 = 0, 0, 0, 0
    for i in range(10):
        train, test, valid = kf(rules, i)
        accuracy, precision, recall, f1 = adacost(train, test, valid)
        print("round-%d,accuracy(a):%f,precision(5):%f,recall(5):%f,f1:%f" %
              (i+1, accuracy, precision, recall, f1))
        sum_acc += 0.1*accuracy
        sum_pre += 0.1*precision
        sum_rec += 0.1*recall
        sum_f1 += 0.1*f1
    print("accuracy(a):%f,precision(5):%f,recall(5):%f,f1:%f" %
          (sum_acc, sum_pre, sum_rec, sum_f1))
