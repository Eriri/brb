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
    one, les = np.ptp(ant, axis=0) * 0.5, list(set(con))
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


def Nabla(a, c, ba, bc, oi):
    '''
    oi = np.random.randint(0, len(ba))
    p = Pool()
    r = [p.apply_async(Nabla, (a, c, ba, bc, oi,)) for a, c in zip(ta[bm], tc[bm])]
    p.close(), p.join()
    n = mu * np.sum([res.get() for res in r], axis=0) / bs
    ba[oi] -= mu * n / bs
    '''
    w = np.exp(- np.sum((ba-a)*(ba-a)/(one*one), axis=1) / 2)
    sw = np.sum(w)
    n = np.zeros(ba[oi].shape)
    if np.max(w) == sw:
        return n
    bt = np.transpose(bc) * w / (sw - w) + 1
    b = np.prod(bt, axis=1) - 1
    pc = b / np.sum(b)
    for j in range(len(les)):
        d1 = (np.sum(b) - b[j]) / (np.sum(b) * np.sum(b))
        bn = - np.transpose(bc)[j] * w / ((sw - w) * (sw - w)) * (np.prod(bt[j]) / bt[j])
        bn[oi] = bc[oi][j] / (sw - w[oi]) * np.prod(bt[j]) / bt[j][oi]
        d2 = np.sum(bn)
        d3 = (a - ba[oi])/(one * one) * np.exp(-np.sum((a - ba[oi])*(a - ba[oi])/(one * one)) / 2)
        n += (pc[j] - c[j]) * d1 * d2 * d3
    return n


def GradientDescent(ta, tc, bs, mu, ba, bc):
    bi = np.arange(len(ta)-len(ta) % bs)
    np.random.shuffle(bi)
    pb = tqdm.tqdm(total=len(bi)/bs)
    for i, bm in enumerate(np.split(bi, len(bi)/bs)):
        #oi  =np.random.randint(0, len(ba))
        oi = i % len(ba)
        t = i % len(ba)
        n = np.zeros(ba[oi].shape)
        for a, c in zip(ta[bm], tc[bm]):
            w = np.exp(- np.sum((ba-a)*(ba-a)/(one*one), axis=1) / 2)
            sw = np.sum(w)
            if np.max(w) == sw:
                continue

            # B = np.transpose(np.exp(bc) / np.sum(np.exp(bc), axis=1)[:, None])  # (N,L)
            B = np.transpose(bc)
            Bi = B * w / (sw - w) + 1  # (N,L)
            B_ = np.prod(Bi, axis=1) - 1  # (N)
            pc = B_ / np.sum(B_)  # (N)
            dBB_ = (np.sum(B_) - B_) / np.sum(B_) / np.sum(B_)  # (N)

            bn1 = - B * w / (sw - w) / (sw - w) * (np.prod(Bi, axis=1)[:, None] / Bi)  # (N,L)
            bn2 = B[:, t] / (sw - w[t]) * np.prod(Bi, axis=1) / Bi[:, t]  # (N)
            dB_a = np.sum(bn1, axis=1) - bn1[:, t] + bn2  # (N)
            dax = (a - ba[t]) / one / one * np.exp(-np.sum((a - ba[t])*(a - ba[t]) / one / one) / 2)  # (T)
            n += np.sum((pc - c) * dBB_ * dB_a) * dax

            '''
            bt = np.transpose(bc) * w / (sw - w) + 1
            b = np.prod(bt, axis=1) - 1
            pc = b / np.sum(b)
            for j in range(len(les)):
                d1 = (np.sum(b) - b[j]) / (np.sum(b) * np.sum(b))
                bn = - np.transpose(bc)[j] * w / ((sw - w) * (sw - w)) * (np.prod(bt[j]) / bt[j])
                bn[oi] = bc[oi][j] / (sw - w[oi]) * np.prod(bt[j]) / bt[j][oi]
                d2 = np.sum(bn)
                d3 = (a - ba[oi])/(one * one) * np.exp(-np.sum((a - ba[oi])*(a - ba[oi])/(one * one)) / 2)
                n += (pc[j] - c[j]) * d1 * d2 * d3
            '''
        ba[oi] -= mu * n / bs
        pb.update()
    pb.close()
    return ba, bc


def main():
    ant, con = ReadData("../data/Skin_NonSkin.txt")
    cnt, bn, zc = 5000, 100, []

    skf = StratifiedKFold(10, True)
    for train_mask, test_mask in skf.split(ant, np.argmax(con, axis=1)):
        train_ant, train_con = ant[train_mask], con[train_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]

        # pred_con=np.array([EvidentialReasoing(a, train_ant, train_con) for a in train_ant])
        # print(accuracy_score(np.argmax(test_con, axis=1), np.argmax(pred_con, axis=1)))

        base_ant = np.random.uniform(low, high, (bn, ant.shape[1]))
        base_con = []
        acc, err = [], []
        for i in range(base_ant.shape[0]):
            bc = np.zeros((train_con.shape[1],))
            bc[i % len(les)] = 1.0
            base_con.append(bc)
        base_con = np.array(base_con)
        # base_con /= np.sum(base_con, axis=1)[:, None]
        sc = 0.0
        for _ in range(cnt):
            base_ant, base_con = GradientDescent(train_ant, train_con, 128, 1e7*(1-sc), base_ant, base_con)
            pred_con = np.array([EvidentialReasoing(a, base_ant, base_con) for a in train_ant])
            error = np.sum((pred_con - train_con) * (pred_con - train_con))
            sc = accuracy_score(np.argmax(train_con, axis=1), np.argmax(pred_con, axis=1))
            pt_con = np.array([EvidentialReasoing(a, base_ant, base_con) for a in test_ant])
            pc = accuracy_score(np.argmax(test_con, axis=1), np.argmax(pt_con, axis=1))
            print(_, error, sc, pc)
            acc.append(sc), err.append(error)
            if sc > 0.935:
                break
        pred_con = np.array([EvidentialReasoing(a, base_ant, base_con) for a in test_ant])
        print(accuracy_score(np.argmax(test_con, axis=1), np.argmax(pred_con, axis=1)))
        view.draw(acc, err)
        zc.append(accuracy_score(np.argmax(test_con, axis=1), np.argmax(pred_con, axis=1)))
    print(zc)


if __name__ == "__main__":
    main()
