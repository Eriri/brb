from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

FlowDiff = np.arange(-10,2,1)
PressureDiff = np.arange(-0.02,0.04,0.01)

x = []
y = []
z = []

f = open("../data/oil_testdata_2007.txt","r")
for line in f.readlines():
    line = str.split(line)
    if len(line) == 3:
        y.append(float(line[0]))
        x.append(float(line[1]))
        z.append(float(line[2]))

x = np.array(x)
y = np.array(y)
z = np.array(z)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z)
ax.set_xlabel("PressureDiff")
ax.set_ylabel("FlowDiff")
ax.set_zlabel("LeakSize")

plt.show()