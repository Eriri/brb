import numpy as np
from numpy import arange, array, zeros
from numpy import exp, amin, amax, sum, ptp, square
from numpy import argmax, prod, transpose, concatenate, split, expand_dims
from numpy.random import uniform, shuffle
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import tqdm
import time
import view

les, one, low, high, eps = arange(8), array([]), array([]), array([]), 1e-8


def trans(t):
    z = zeros((len(les),))
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
    one = ptp(ant, axis=0) * 0.1
    low, high = amin(ant, axis=0), amax(ant, axis=0)
    con = array([trans(x) for x in con])
    return array(ant), array(con)


def EvidentialReasoing(a, ant, con, rw, tau):
    w = exp(-0.5*sum(tau*(ant-a)*(ant-a)/one/one, axis=1))*rw
    sw, b = sum(w), uniform(size=len(les))
    if amax(w) == sw:
        return [b/sum(b), con[argmax(w)]][int(sw != 0.0)]
    b = prod(transpose(con)*w/(sw-w)+1, axis=1)-1
    return b/sum(b)


def AdamOptimize(ta, tc, ep, bs, ba, bc, bw, bt):
    # ba (L,T) bc (L,N) bw (L)
    while len(ta) % bs != 0:
        ta = concatenate((ta, ta[:len(ta) % bs]))
        tc = concatenate((tc, tc[:len(tc) % bs]))
    bi, bn = arange(len(ta)), len(ta)/bs
    pb = tqdm.tqdm(total=bn)
    for epi in range(ep):
        shuffle(bi)
        for mask in split(bi, len(bi)/bs):
            ant, con = ta[bi[mask]], tc[bi[mask]]  # (bs,T) (bs,N)
            nx, ny, nz = zeros(ba.shape), zeros(bc.shape), zeros(bw.shape)
            theta, tau = 1/(1+exp(-bw)), 1/(1+exp(-bt))  # (L) (T)
            alpha = exp(-sum(tau*square((ant[:, None]-ba)/one), axis=2))  # (bs,L)
            w = alpha*theta+eps  # (bs,L)
            sw = sum(w, axis=1)  # (bs)
            B = transpose(exp(bc)/sum(exp(bc), axis=1)[:, None])  # (N,L)
            Bi = B*w[:, None]/(sw[:, None]-w)[:, None]+1  # (bs,N,L)
            B_ = prod(Bi, axis=2) - 1  # (bs,N)
            pc = B_/sum(B_, axis=1)[:, None]  # (bs,N)
            dBB_ = (sum(B_, axis=1)[:, None]-B_)/sum(B_, axis=1)[:, None]/sum(B_, axis=1)[:, None]  # (bs,N)
            T1 = theta*B*w/(sw-w)/(sw-w)*(prod(Bi)/Bi)
            print(T1.shape)
            exit(0)
        pb.update()
    pb.clear()
    return ba, bc, bw, bt


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


def Evaluating(ant, con, ba, bc, bw, bt):
    bc = np.exp(bc)/np.sum(np.exp(bc), axis=1)[:, None]
    bw, bt = 1/(1+exp(-bw)), 1/(1+exp(-bt))
    pc = np.array([EvidentialReasoing(a, ba, bc, bw, bt) for a in ant])
    res = {}
    res['err'] = sum((con - pc)*(con - pc))
    res['acc'] = accuracy_score(np.argmax(con, axis=1), argmax(pc, axis=1))
    oc = sum(con * les, axis=1)
    op = sum(pc * les, axis=1)
    res['mse'] = sum((oc - op) * (oc - op)) / len(con)
    res['mea'] = sum(np.abs(oc - op)) / len(con)
    return res


def main():
    ant, con = ReadData("../data/oil_rev.txt")
    bn = 50

    # kf = StratifiedKFold(10, True)
    # for train_mask, test_mask in kf.split(ant, np.argmax(con, axis=1)):

    kf = KFold(4, shuffle=True)
    for train_mask, test_mask in kf.split(ant):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]

        base_ant = uniform(low, high, (bn, ant.shape[1],))
        base_con = uniform(-1.0, 1.0, (bn, con.shape[1],))
        base_wei = uniform(-1.0, 1.0, (bn,))
        base_tau = uniform(-1.0, 1.0, (ant.shape[1],))

        base_ant, base_con, base_wei, base_tau = AdamOptimize(train_ant, train_con, 10, 64,
                                                              base_ant, base_con, base_wei, base_tau)

        print(Evaluating(train_ant, train_con, base_ant, base_con, base_wei))
        print(Evaluating(test_ant, test_con, base_ant, base_con, base_wei))

        # for _ in range(cnt):
        #     base_ant, base_con, base_wei = GradientDescent(train_ant, train_con, 16,
        #                                                    1e-3*(1-sc), 1e2*(1-sc), 1e1*(1-sc),
        #                                                    base_ant, base_con, base_wei)

        #     a1, e1, mse1, mae1 = Evaluating(train_ant, train_con, base_ant, base_con, base_wei)
        #     a2, e2, mse2, mae2 = Evaluating(test_ant, test_con, base_ant, base_con, base_wei)
        #     print(_, a1, e1, mse1, mae1)
        #     print(_, a2, e2, mse2, mae2)
        #     if mae1 < 0.1:
        #         base_con = np.exp(base_con) / np.sum(np.exp(base_con), axis=1)[:, None]
        #         view.draw3d_compare(train_ant[:, 0], train_ant[:, 1], np.sum(train_con * les, axis=1),
        #                             base_ant[:, 0], base_ant[:, 1], np.sum(base_con * les, axis=1))


if __name__ == "__main__":
    main()
