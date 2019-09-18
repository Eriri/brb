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


def EvidentialReasoing(ant, con, a):
    w = np.prod(np.exp(-0.5*(ant-a)*(ant-a)/one/one), axis=1)
    sw, b = np.sum(w), np.random.uniform(size=len(les))
    if np.max(w) == sw:
        return [b/np.sum(b), con[np.argmax(w)]][sw != 0.0]
    for i in range(les.shape[0]):
        b[i] = np.prod((con[:, i] * w) / (sw - w) + 1)
    return (b - 1)/(np.sum(b) - len(les))


def StochasticGradientDescent(ba, bc, ta, tc, oi, cnt):
    for _ in range(cnt):

        pc = np.array([EvidentialReasoing(ba, bc, a) for a in ta])
        error = 0.5 * (tc - pc)*(tc - pc)
        print(error)


def main():
    ant, con = ReadData("../data/iris.data", ',')
    print(ant, con)
    print(les, one)


if __name__ == "__main__":
    main()
