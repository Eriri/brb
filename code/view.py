from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random


def read(filename, c, limit=None):
    ox, oy, oz = [], [], []
    with open(filename) as f:
        for line in f.readlines():
            line = str.split(line)
            if len(line) == 4 and int(line[3]) == c:
                ox.append(float(line[0]))
                oy.append(float(line[1]))
                oz.append(float(line[2]))
    if limit is not None:
        c = list(zip(ox, oy, oz))
        random.shuffle(c)
        ox[:], oy[:], oz[:] = zip(*c)
        return np.array(ox)[:limit], np.array(oy)[:limit], np.array(oz)[:limit]
    return np.array(ox), np.array(oy), np.array(oz)


def draw(acc, err):
    x = np.arange(len(acc))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, acc, 'b')
    ax1.set_ylabel('accuracy')
    ax2 = ax1.twinx()
    ax2.plot(x, err, 'r')
    ax2.set_ylabel('error')
    plt.show()

    pass


def draw3d():
    fig = plt.figure()
    a = Axes3D(fig)
    ox, oy, oz = read("../data/Skin_NonSkin.txt", 1, 2500)
    a.scatter(ox, oy, oz, c='b')
    tx, ty, tz = read("../data/Skin_NonSkin.txt", 2, 2500)
    a.scatter(tx, ty, tz, c='r')
    plt.show()


def draw3d_compare(ox, oy, oz, tx, ty, tz):
    fig = plt.figure()
    a = Axes3D(fig)
    a.scatter(ox, oy, oz, c='b')
    a.scatter(tx, ty, tz, c='r')
    plt.show()
