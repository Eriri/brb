import numba as nb
import numpy as np
from numba import cuda

xs = (4, 4)


@cuda.jit
def kernel(res):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    st = cuda.shared.array(xs, nb.float64)
    st[x][y] = x + y * cuda.blockDim.x
    cuda.syncthreads()
    if x == 0 and y == 0:
        for i in range(4):
            for j in range(4):
                res[i][j] = st[i][j]


a = np.ones(10, np.float)
b = np.ones(10, np.float)
c = np.dot(a, b)
print(c)
