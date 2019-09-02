import numpy as np
from ADASYN import ADASYN
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from multiprocessing import Pool
import tqdm


def ReadData(filename):
    ant, con = [], []
    for i in open(file=filename, mode="r"):
        e = list(map(float, i.split()))
        ant.append(e[:-1]), con.append(e[-1])
    ant, con = np.array(ant), np.array(con)
    # ant = PCA().fit_transform(ant, con)
    for i in range(con.shape[0]):
        if con[i] != 1.0:
            con[i] = 0.0
    one = np.ptp(ant, axis=0) * 0.01
    les = np.array(list(set(con)))
    return ant, con, one, les


def EvidentialReasoing(ant, con, one, les, rw, a):
    wei = np.prod(np.exp(-np.abs((ant-a)/one)), axis=1) * rw
    sw = np.sum(wei)
    if sw == 0.0:
        return np.random.randint(0, les.shape[0])
    wei = wei / sw
    if len(wei[wei > 1e-6]) > 400:
        print(len(wei[wei > 1e-6]))
        print("ops")
        exit(0)
    md = 1 - wei
    if np.min(md) == 0.0:
        return int(con[np.argmin(md)])
    b = np.empty(shape=les.shape)
    for i in range(les.shape[0]):
        b[i] = np.prod(wei[con == les[i]] / md[con == les[i]] + 1)
    s = np.sum(b)
    b = (b - 1)/(s - les.shape[0])
    return np.argmax(b)


def GenerateBase(ant, con, one, les, cnt, bn, lo):
    skf = StratifiedKFold(n_splits=bn)
    adsn = ADASYN(k=2, imb_threshold=0.6, ratio=1.0, verbose=False)
    pb = tqdm.tqdm(total=cnt*bn, position=lo)
    base = []
    for base_mask, test_mask in skf.split(ant, con):
        base_ant, base_con = ant[base_mask], con[base_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]
        base_ant, base_con = adsn.fit_transform(base_ant, base_con)
        best, bw = 0.0, np.empty(base_ant.shape[0])
        for i in range(cnt):
            rw = 1.0 - np.random.rand(base_ant.shape[0])
            for ta, tc in zip(test_ant, test_con):
                if les[EvidentialReasoing(base_ant, base_con, one, les, rw, ta)] != tc:
                    tw = np.prod(np.exp(-np.abs((base_ant-ta)/one)), axis=1) * rw
                    tw /= np.max(tw) * 4.0
                    if tc == 1.0:
                        rw[base_con == 0.0] /= 1.0 + tw[base_con == 0.0]
                    if tc == 0.0:
                        rw[base_con == 0.0] *= 1.0 + tw[base_con == 0.0]
            pred_con = Evaluating([[base_ant, base_con, rw]], test_ant, one, les)
            c = f1_score(1.0-test_con, 1.0-pred_con)
            if c > best:
                best, bw = c, rw
            pb.update()
        base.append([base_ant, base_con, bw])
    pb.close()
    return base


def Evaluating(base, test_ant, one, les, p=0):
    test_con = []
    for ta in test_ant:
        tc = [0, 0]
        for base_ant, base_con, rw in base:
            tc[EvidentialReasoing(base_ant, base_con, one, les, rw, ta)] += 1
        test_con.append(tc.index(max(tc)))
    return np.array(test_con)


def Ebrb(train_ant, train_con, test_ant, test_con, one, les):
    pred_con = []
    for ta in test_ant:
        pred_con.append(EvidentialReasoing(train_ant, train_con, one, les, 1.0, ta))
    pred_con = np.array(pred_con)
    return Result(test_con, pred_con)


def Result(test_con, pred_con):
    res = {}
    res['accuracy'] = accuracy_score(test_con, pred_con)
    res['precision'] = precision_score(1.0-test_con, 1.0-pred_con)
    res['recall'] = recall_score(1.0-test_con, 1.0-pred_con)
    res['f1'] = f1_score(1.0-test_con, 1.0-pred_con)
    res['g-mean'] = geometric_mean_score(1.0-test_con, 1.0-pred_con)
    return res


def Train(ant, con, one, les, mask, cnt, bn, lo):
    train_mask, test_mask = mask
    train_ant, train_con = ant[train_mask], con[train_mask]
    test_ant, test_con = ant[test_mask], con[test_mask]
    base = GenerateBase(train_ant, train_con, one, les, cnt, bn, lo)
    pred_con = Evaluating(base, test_ant, one, les, 1)
    return Result(test_con, pred_con)


def main():
    ant, con, one, les = ReadData("page-blocks.data")
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    eva = ['accuracy', 'precision', 'recall', 'f1', 'g-mean']

    # D = dict([(k, 0) for k in eva])
    # p = Pool(10)
    # cnt = 10  # 迭代次数
    # bn = 5  # 基学习器个数
    # res = [p.apply_async(Train, (ant, con, one, les, mask, cnt, bn, l,)) for l, mask in enumerate(skf.split(ant, con))]
    # p.close(), p.join()
    # res = [r.get() for r in res]
    # for d in res:
    #     print()
    #     for k in eva:
    #         D[k] += 0.1 * d[k]
    # print(D)

    for mask in skf.split(ant, con):
        print(Train(ant, con, one, les, mask, 1, 3, 0))
        break

    # E = dict([(k, 0) for k in eva])
    # p = Pool(10)
    # res = [p.apply_async(Ebrb, (ant[t1], con[t1], ant[t2], con[t2], one, les)) for t1, t2 in skf.split(ant, con)]
    # p.close(), p.join()
    # res = [r.get() for r in res]
    # for d in res:
    #     for k in eva:
    #         E[k] += 0.1 * d[k]
    # print(E)


if __name__ == "__main__":
    main()
