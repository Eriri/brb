import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import tqdm
import time
import view

les, one, low, high = np.arange(8), np.array([]), np.array([]), np.array([])


def trans(t):
    z = np.zeros((len(les),))
    for i in range(len(les)):
        if t == les[i]:
            z[i] = 1.0
            return z
    for i in range(len(les)-1):
        if t > les[i] and t < les[i+1]:
            z[i] = les[i+1] - t
            z[i+1] = t - les[i]
            return z


def ReadData(filename):
    ant, con = [], []
    with open(file=filename, mode='r') as f:
        for i in f:
            e = list(map(float, i.split()))
            ant.append(e[:-1]), con.append(e[-1])
    global one, les, low, high
    # one, les = np.ptp(ant, axis=0) * 0.1, list(set(con))
    one = np.ptp(ant, axis=0) * 0.1
    low, high = np.min(ant, axis=0), np.max(ant, axis=0)
    con = np.array([trans(x) for x in con])
    # for i in range(len(con)):
    # o = np.zeros((len(les),))
    # o[les.index(con[i])] = 1.0
    # con[i] = o
    return np.array(ant), np.array(con)


def EvidentialReasoing(a, ant, con, rw):
    w = np.exp(-0.5 * np.sum((ant-a)*(ant-a)/(one*one), axis=1)) * rw
    sw, b = np.sum(w), np.random.uniform(size=len(les))
    if np.max(w) == sw:
        return [b/np.sum(b), con[np.argmax(w)]][int(sw != 0.0)]
    b = np.prod(np.transpose(con) * w / (sw - w) + 1, axis=1) - 1
    return b / np.sum(b)


def AdamOptimize(ta, tc, ep, bs, ba, bc, bw):
    # ba (L,T) bc (L,N) bw (L)
    bi = np.arange(len(ta))
    for epi in range(ep):
        np.random.shuffle(bi)
        for i in range(np.ceil(len(bi) / bs)):
            ant, con = ta[bi[i*bs:i*bs+bs]], tc[bi[i*bs:i*bs+bs]]  # (bs,T) (bs,N)
            nx, ny, nz = np.zeros(ba.shape), np.zeros(bc.shape), np.zeros(bw.shape)
            theta = 1 / (1 + np.exp(-bw))  # (L)
            alpha = np.exp(-np.sum((ant-ba)*(ant-ba)/one/one/2, axis=1))


def GradientDescent(ta, tc, ep, bs, ba, bc, bw):
    bi = np.arange(len(ta)-len(ta) % bs)
    np.random.shuffle(bi)
    pb = tqdm.tqdm(total=len(bi)/bs)
    for i, bm in enumerate(np.split(bi, len(bi)/bs)):
        t = i % len(ba)
        # t = np.random.randint(0, len(ba))
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
    oc = np.sum(con * les, axis=1)
    op = np.sum(pc * les, axis=1)
    mse = np.sum((oc - op) * (oc - op)) / len(con)
    mae = np.sum(np.abs(oc - op)) / len(con)
    return acc, err, mse, mae


def main():
    ant, con = ReadData("../data/oil_rev.txt")
    cnt, bn = 5000, 50

    # kf = StratifiedKFold(10, True)
    # for train_mask, test_mask in kf.split(ant, np.argmax(con, axis=1)):

    kf = KFold(4, shuffle=True)
    for train_mask, test_mask in kf.split(ant):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]

        # pred_con = np.array([EvidentialReasoing(a, train_ant, train_con, 1.0) for a in train_ant])
        # print(accuracy_score(np.argmax(test_con, axis=1), np.argmax(pred_con, axis=1)))

        base_ant = np.random.uniform(low, high, (bn, ant.shape[1]))
        base_con = np.random.uniform(-1.0, 1.0, (bn, con.shape[1]))
        base_wei = np.random.uniform(-1.0, 1.0, (bn,))

        # base_con = np.zeros((bn, con.shape[1]))
        # for i in range(len(base_con)):
        #     base_con[i][i % 8] = 10.0
        # base_wei = np.zeros((bn,))

        # acc, err = [], []
        sc = 0.0
        for _ in range(cnt):
            base_ant, base_con, base_wei = GradientDescent(train_ant, train_con, 16,
                                                           1e-3*(1-sc), 1e2*(1-sc), 1e1*(1-sc),
                                                           base_ant, base_con, base_wei)

            a1, e1, mse1, mae1 = Evaluating(train_ant, train_con, base_ant, base_con, base_wei)
            a2, e2, mse2, mae2 = Evaluating(test_ant, test_con, base_ant, base_con, base_wei)
            print(_, a1, e1, mse1, mae1)
            print(_, a2, e2, mse2, mae2)
            if mae1 < 0.1:
                base_con = np.exp(base_con) / np.sum(np.exp(base_con), axis=1)[:, None]
                view.draw3d_compare(train_ant[:, 0], train_ant[:, 1], np.sum(train_con * les, axis=1),
                                    base_ant[:, 0], base_ant[:, 1], np.sum(base_con * les, axis=1))
                exit(0)


if __name__ == "__main__":
    main()
