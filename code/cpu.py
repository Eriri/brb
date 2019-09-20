import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import tqdm
import time

les, one, low, high = [], np.array([]), np.array([]), np.array([])


def ReadData(filename):
    ant, con = [], []
    with open(file=filename, mode='r') as f:
        for i in f:
            e = list(map(float, i.split()))
            ant.append(e[:-1]), con.append(e[-1])
    global one, les, low, high
    one, les = np.ptp(ant, axis=0) * 0.1, list(set(con))
    low, high = np.min(ant, axis=0), np.max(ant, axis=0)
    for i in range(len(con)):
        o = np.zeros((len(les),))
        o[les.index(con[i])] = 1.0
        con[i] = o
    return np.array(ant), np.array(con)


def EvidentialReasoing(a, ant, con):
    w = np.exp(-0.5 * np.sum((ant-a)*(ant-a)/(one*one), axis=1))
    sw, b = np.sum(w), np.random.uniform(size=len(les))
    if np.max(w) == sw:
        return [b/np.sum(b), con[np.argmax(w)]][int(sw != 0.0)]
    b = np.prod(np.transpose(con) * w / (sw - w) + 1, axis=1) - 1
    return b / np.sum(b)


def GradientDescent(ta, tc, bs, mu, ba, bc):
    bi = np.arange(len(ta)-len(ta) % bs)
    np.random.shuffle(bi)
    for bm in np.split(bi, len(bi)/bs):
        oi = np.random.randint(0, len(ba))
        n = np.zeros(ba[oi].shape)
        for a, c in zip(ta[bm], tc[bm]):
            w = np.exp(- np.sum((ba-a)*(ba-a)/(one*one), axis=1) / 2)
            sw = np.sum(w)
            if np.max(w) == sw:
                continue
            bt = np.transpose(bc) * w / (sw - w) + 1
            b = np.prod(bt, axis=1) - 1
            pc = b / np.sum(b)
            for j in range(len(les)):
                d1 = - (np.sum(b) - b[j]) / (np.sum(b) * np.sum(b))
                bn = - np.transpose(bc)[j] * w / ((sw - w) * (sw - w)) * (np.prod(bt[j]) / bt[j])
                bn[oi] = bc[oi][j] / (sw - w[oi]) * np.prod(bt[j]) / bt[j][oi]
                d2 = np.sum(bn)
                d3 = (a - ba[oi])/(one * one) * np.exp(-np.sum((a - ba[oi])*(a - ba[oi])/(one * one)) / 2)
                n += (pc[j] - c[j]) * d1 * d2 * d3
        ba[oi] += mu * n / bs
    return ba, bc


def main():
    ant, con = ReadData("../data/iris_rev.data")
    cnt, bn, zc = 1000, 20, []

    skf = StratifiedKFold(5, True)
    for train_mask, test_mask in skf.split(ant, np.argmax(con, axis=1)):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]
        base_ant = np.random.uniform(low, high, (bn, ant.shape[1]))
        base_con = []
        for i in range(base_ant.shape[0]):
            bc = np.zeros((train_con.shape[1],))
            bc[i % len(les)] = 1.0
            base_con.append(bc)
        base_con = np.array(base_con)
        # base_con /= np.sum(base_con, axis=1)[:, None]
        sc = 0.0
        for _ in range(cnt):
            base_ant, base_con = GradientDescent(train_ant, train_con, 16, 5*(1.0-sc), base_ant, base_con)
            pred_con = np.array([EvidentialReasoing(a, base_ant, base_con) for a in train_ant])
            error = np.sum((pred_con - train_con) * (pred_con - train_con))
            sc = accuracy_score(np.argmax(train_con, axis=1), np.argmax(pred_con, axis=1))
            print(error, sc)
            if error < 0.1 * len(train_con) or sc > 0.95:
                pred_con = np.array([EvidentialReasoing(a, base_ant, base_con) for a in test_ant])
                print(accuracy_score(np.argmax(test_con, axis=1), np.argmax(pred_con, axis=1)))
                zc.append(accuracy_score(np.argmax(test_con, axis=1), np.argmax(pred_con, axis=1)))
                break
    print(zc)


if __name__ == "__main__":
    main()
