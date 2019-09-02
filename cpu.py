import numpy as np
from ADASYN import ADASYN
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
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
    md = 1 - wei
    if np.min(md) == 0.0:
        return int(con[np.argmin(md)])
    b = np.empty(shape=les.shape)
    for i in range(les.shape[0]):
        b[i] = np.prod(wei[con == les[i]] / md[con == les[i]] + 1)
    s = np.sum(b)
    b = (b - 1)/(s - les.shape[0])
    return np.argmax(b)


def GenerateBase(ant, con, one, les,  cnt, bn, lo):
    skf = StratifiedKFold(n_splits=bn)
    adsn = ADASYN(k=2, imb_threshold=0.6, ratio=0.5, verbose=False)
    pb = tqdm.tqdm(total=cnt*bn, position=lo)
    base = []
    for base_mask, test_mask in skf.split(ant, con):
        base_ant, base_con = ant[base_mask], con[base_mask]
        test_ant, test_con = ant[test_mask], con[test_mask]
        adsn.fit(base_ant, base_con)
        good_ant = []
        need_size = int(2.0 * np.sum(base_con == 0.0))
        for i in range(cnt):
            new_ant, new_con = adsn.oversample()
            all_ant = np.concatenate([new_ant, base_ant[base_con == 1.0]])
            all_con = np.concatenate([new_con, base_con[base_con == 1.0]])
            new_size, rw = len(new_ant), np.ones((len(all_ant),))
            for ta in test_ant[test_con == 1.0]:
                if EvidentialReasoing(all_ant, all_con, one, les, 1.0, ta) != 1:
                    tw = np.prod(np.exp(-np.abs((all_ant-ta)/one)), axis=1)
                    rw[:new_size][tw[:new_size] > np.max(tw[:new_size])] = 0
            good_ant.append(new_ant[rw[:new_size] == 1])
            pb.update()
        good_ant = np.concatenate(good_ant)
        need_size = min(need_size, len(good_ant))
        best = 0.0
        for i in range(cnt):
            np.random.shuffle(good_ant)
            all_ant = np.concatenate([base_ant, good_ant[:need_size]])
            all_con = np.concatenate([base_con, np.zeros((need_size,))])
            fs = f1_score(1.0-test_con, 1.0-Evaluating(
                [[all_ant, all_con, 1.0]], test_ant, one, les))
            if best < fs:
                best, best_ant, best_con = fs, all_ant, all_con
        base.append([best_ant, best_con, 1.0])
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


def Ebrb(ant, con, one, les, mask):
    train_mask, test_mask = mask
    train_ant, train_con = ant[train_mask], con[train_mask]
    test_ant, test_con = ant[test_mask], con[test_mask]
    pred_con = []
    for ta in test_ant:
        pred_con.append(EvidentialReasoing(
            train_ant, train_con, one, les, 1.0, ta))
    pred_con = np.array(pred_con)
    return Result(test_con, pred_con)


def Train(ant, con, one, les, mask, cnt, bn, lo):
    train_mask, test_mask = mask
    train_ant, train_con = ant[train_mask], con[train_mask]
    test_ant, test_con = ant[test_mask], con[test_mask]
    base = GenerateBase(train_ant, train_con, one, les, cnt, bn, lo)
    pred_con = Evaluating(base, test_ant, one, les, 1)
    return Result(test_con, pred_con)


def Result(test_con, pred_con):
    res = {}
    res['accuracy'] = accuracy_score(test_con, pred_con)
    res['precision'] = precision_score(1.0-test_con, 1.0-pred_con)
    res['recall'] = recall_score(1.0-test_con, 1.0-pred_con)
    res['f1'] = f1_score(1.0-test_con, 1.0-pred_con)
    res['g-mean'] = geometric_mean_score(1.0-test_con, 1.0-pred_con)
    return res


def main():
    ant, con, one, les = ReadData("page-blocks.data")
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    eva = ['accuracy', 'precision', 'recall', 'f1', 'g-mean']

    D = dict([(k, 0) for k in eva])
    p = Pool(10)
    cnt = 10  # 迭代次数
    bn = 5  # 基学习器个数
    res = [p.apply_async(Train, (ant, con, one, les, mask, cnt, bn, l,))
           for l, mask in enumerate(skf.split(ant, con))]
    p.close(), p.join()
    res = [r.get() for r in res]
    for d in res:
        print()
        for k in eva:
            D[k] += 0.1 * d[k]
    print(D)
    # E = dict([(k, 0) for k in eva])
    # for i in range(100):
    #     p = Pool(10)
    #     res = [p.apply_async(Ebrb, (ant, con, one, les, mask))
    #            for mask in skf.split(ant, con)]
    #     p.close(), p.join()
    #     res = [r.get() for r in res]
    #     for d in res:
    #         for k in eva:
    #             E[k] += 0.001 * d[k]
    # print(E)

    # for mask in skf.split(ant, con):
    #     print(Train(ant, con, one, les, mask, 10, 3, 0))
    #     print(Ebrb(ant, con, one, les, mask))
    #     break


if __name__ == "__main__":
    main()
