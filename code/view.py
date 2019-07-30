from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

FlowDiff = np.arange(-10,2,1)
PressureDiff = np.arange(-0.02,0.04,0.01)

x = []
y = []
z1 = []
z2 = []

f = open("result","r")
for line in f.readlines():
    line = str.split(line)
    if len(line) == 4:
        y.append(float(line[0]))
        x.append(float(line[1]))
        z1.append(float(line[2]))
        z2.append(float(line[3]))

x = np.array(x)
y = np.array(y)
z1 = np.array(z1)
z2 = np.array(z2)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z1,c='b')
ax.scatter(x,y,z2,c='r')
ax.set_xlabel("PressureDiff")
ax.set_ylabel("FlowDiff")
ax.set_zlabel("LeakSize")

plt.show()