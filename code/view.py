from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def read(filename):
    ox, oy, oz = [], [], []
    with open(filename) as f:
        for line in f.readlines():
            line = str.split(line)
            if len(line) == 3:
                ox.append(float(line[0]))
                oy.append(float(line[1]))
                oz.append(float(line[2]))
    return np.array(ox), np.array(oy), np.array(oz)


fig = plt.figure()
a = Axes3D(fig)
ox, oy, oz = read("./data/oil_rev.txt")
tx, ty, tz = read("./result")
a.scatter(ox, oy, oz, c='b')
a.scatter(tx, ty, tz, c='r')
a.set_xlabel("FlowDiff")
a.set_ylabel("PressureDiff")
a.set_zlabel("LeakSize")
plt.show()
