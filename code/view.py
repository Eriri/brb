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


def draw(x, y1, l1, y2, l2):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, label=l1)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, label=l2)
    ax1.legend()
    ax2.legend()
    # plt.legend()
    plt.show()


def draw3d(x, y, z):
    fig = plt.figure()
    a = Axes3D(fig)
    a.scatter(x, y, z, c='r')
    plt.show()


def draw3d_compare(ox, oy, oz, tx, ty, tz):
    fig = plt.figure()
    a = Axes3D(fig)
    a.scatter(ox, oy, oz, c='b')
    a.scatter(tx, ty, tz, c='r')
    plt.show()


def draw2d(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.show()


def draw2d2(ax, ay, bx, by):
    plt.figure()
    plt.plot(ax, ay, 'r')
    plt.plot(bx, by, 'b')
    plt.show()


def draw2d3(x, y, X, Y):
    plt.figure()
    plt.plot(x, y)
    plt.plot(X, Y, 'ro')
    plt.show()


def drawxy(xy):
    '''[(x,y,label),(x,y,label)...(x,y,label)]'''
    plt.figure()
    for x, y, l in xy:
        plt.plot(x, y, label=l)
    plt.legend()
    plt.show()
