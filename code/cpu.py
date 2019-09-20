import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import tqdm

les, one = np.array([]), []


def ReadData(filename, sep=None):
    global one, les
    ant, con = [], []
    with open(file=filename, mode='r') as f:
        for i in f:
            e = i.split(sep)
            ant.append(list(map(float, e[:-1])))
            con.append(e[-1].strip())
    one = np.ptp(ant, axis=0) * 0.1
    les = list(set(con))
    for i in range(len(con)):
        t = les.index(con[i])
        con[i] = [0] * len(les)
        con[i][t] = 1
    return np.array(ant), np.array(con)


def EvidentialReasoing(ant, con, a, rw=1.0):
    w = np.prod(np.exp(-0.5*(ant-a)*(ant-a)/one/one), axis=1)*rw
    sw, b = np.sum(w), np.random.uniform(size=len(les))
    if np.max(w) == sw:
        return [b/np.sum(b), con[np.argmax(w)]][sw != 0.0]
    for i in range(les.shape[0]):
        b[i] = np.prod((con[:, i] * w) / (sw - w) + 1)
    return (b - 1)/(np.sum(b) - len(les))


def GradientDescent(ba, bc, ta, tc, oi, bs):
    bi = np.arange(len(ta)-len(ta) % bs)
    np.random.shuffle(bi)
    for bm in np.split(bi, bs):
        for a, c in zip(ta[bm], tc[bm]):
            w = np.prod(np.exp(-0.5*(ba-a)*(ba-a)/one/one), axis=1)

    pc = np.array([EvidentialReasoing(ba, bc, a) for a in ta])
    error = 0.5 * (tc - pc)*(tc - pc)


def main():
    ant, con = ReadData("../data/iris.data", ',')


if __name__ == "__main__":
    main()
