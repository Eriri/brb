import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def ReadData(filename):
    ant, con = [], []
    for i in open(file=filename, mode="r"):
        e = list(map(float, i.split()))
        ant.append(e[:-1]), con.append(e[-1])
    ant, con = np.array(ant), np.array(con)
    for i in range(con.shape[0]):
        if con[i] != 1.0:
            con[i] = 0.0
    one = np.ptp(ant, axis=0) * 0.1
    les = np.array(list(set(con)))
    return ant, con, one, les


ant, con, one, les = ReadData("page-blocks.data")
print(one)
# plt.hist(ant[:, 9][con == 1.0], 1000), plt.show()
# plt.hist(ant[:, 9][con == 0.0], 1000), plt.show()

# fig = plt.figure()
# a = Axes3D(fig)

# ox = ant[con == 1.0][:, 0]
# oy = ant[con == 1.0][:, 1]
# oz = ant[con == 1.0][:, 2]
# a.scatter(ox, oy, oz, c='b')

# tx = ant[con == 0.0][:, 0]
# ty = ant[con == 0.0][:, 1]
# tz = ant[con == 0.0][:, 2]
# a.scatter(tx, ty, tz, c='r')

# plt.show()
