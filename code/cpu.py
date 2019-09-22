import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import tqdm
import time
import view

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


def EvidentialReasoing(a, ant, con, rw):
    w = np.exp(-0.5 * np.sum((ant-a)*(ant-a)/(one*one), axis=1)) * rw
    sw, b = np.sum(w), np.random.uniform(size=len(les))
    if np.max(w) == sw:
        return [b/np.sum(b), con[np.argmax(w)]][int(sw != 0.0)]
    b = np.prod(np.transpose(con) * w / (sw - w) + 1, axis=1) - 1
    return b / np.sum(b)


def GradientDescent(ta, tc, bs, mua, muc, muw, ba, bc, bw):
    bi = np.arange(len(ta)-len(ta) % bs)
    np.random.shuffle(bi)
    pb = tqdm.tqdm(total=len(bi)/bs)
    for i, bm in enumerate(np.split(bi, len(bi)/bs)):
        t = i % len(ba)
        nx, ny, nz = np.zeros(ba[t].shape), np.zeros(bc[t].shape), 0.0  # (T),(N),()
        for a, c in zip(ta[bm], tc[bm]):
            theta = 1 / (1 + np.exp(-bw))  # (L)
            alpha = np.exp(-np.sum((ba-a)*(ba-a)/(one*one), axis=1) / 2)  # (L)
            w = alpha * theta  # (L)
            sw = np.sum(w)
            if np.max(w) == sw:
                continue
            B = np.transpose(np.exp(bc) / np.sum(np.exp(bc), axis=1)[:, None])  # (N,L)
            Bi = B * w / (sw - w) + 1  # (N,L)
            B_ = np.prod(Bi, axis=1) - 1  # (N)
            pc = B_ / np.sum(B_)  # (N)
            dBB_ = (np.sum(B_) - B_) / (np.sum(B_) * np.sum(B_))  # (N)

            bn1 = - theta[t] * B * w / (sw - w) / (sw - w) * (np.prod(Bi, axis=1)[:, None] / Bi)  # (N,L)
            bn2 = theta[t] * B[:, t] / (sw - w[t]) * np.prod(Bi, axis=1) / Bi[:, t]  # (N)
            dB_a = np.sum(bn1, axis=1) - bn1[:, t] + bn2  # (N)
            dax = (a - ba[t]) / one / one * np.exp(-np.sum((a - ba[t])*(a - ba[t]) / one / one) / 2)  # (T)
            nx += np.sum((pc - c) * dBB_ * dB_a) * dax

            dB_b = w[t] / (sw - w[t]) * np.prod(Bi, axis=1) / Bi[:, t]  # (N)
            dby = np.exp(bc[t]) * (np.sum(np.exp(bc[t])) - np.exp(bc[t])) / (np.sum(np.exp(bc[t])) * np.sum(np.exp(bc[t])))  # (N)
            ny += (pc - c) * dBB_ * dB_b * dby

            bn3 = - alpha[t] * B * w / ((sw - w) * (sw - w)) * (np.prod(Bi, axis=1)[:, None] / Bi)  # (N,L)
            bn4 = alpha[t] * B[:, t] / (sw - w[t]) * np.prod(Bi, axis=1) / Bi[:, t]  # (N)
            dB_t = np.sum(bn3, axis=1) - bn3[:, t] + bn4  # (N)
            dtz = theta[t] * (1.0 - theta[t])
            nz += np.sum((pc - c) * dBB_ * dB_t * dtz)

        ba[t] -= mua * nx / bs
        bc[t] -= muc * ny / bs
        bw[t] -= muw * nz / bs
        pb.update()
    pb.close()
    return ba, bc, bw


def Evaluating(ant, con, ba, bc, bw):
    bc = np.exp(bc) / np.sum(np.exp(bc), axis=1)[:, None]
    bw = 1 / (1 + np.exp(-bw))
    pc = np.array([EvidentialReasoing(a, ba, bc, bw) for a in ant])
    err = np.sum((con - pc)*(con - pc))
    acc = accuracy_score(np.argmax(con, axis=1), np.argmax(pc, axis=1))
    return acc, err


def main():
    ant, con = ReadData("../data/glass_rev.data")
    cnt, bn = 5000, 50

    skf = StratifiedKFold(10, True)
    for train_mask, test_mask in skf.split(ant, np.argmax(con, axis=1)):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]

        # pred_con=np.array([EvidentialReasoing(a, train_ant, train_con) for a in train_ant])
        # print(accuracy_score(np.argmax(test_con, axis=1), np.argmax(pred_con, axis=1)))

        base_ant = np.random.uniform(low, high, (bn, ant.shape[1]))
        base_con = np.random.uniform(-1.0, 1.0, (bn, con.shape[1]))
        base_wei = np.random.uniform(-1.0, 1.0, (bn,))

        # acc, err = [], []
        sc = 0.0
        for _ in range(cnt):
            base_ant, base_con, base_wei = GradientDescent(train_ant, train_con, 8,
                                                           1e-3*(1-sc), 1e-3*(1-sc), 1e-3*(1-sc),
                                                           base_ant, base_con, base_wei)

            a1, e1 = Evaluating(train_ant, train_con, base_ant, base_con, base_wei)
            a2, e2 = Evaluating(test_ant, test_con, base_ant, base_con, base_wei)
            print(a1, e1, a2, e2)


if __name__ == "__main__":
    main()
