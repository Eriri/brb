import numpy as np
import numba as nb
from math import exp, sqrt, erf
from numba import cuda, jit


def ReadData(filename):
    ant, con = [], []
    for i in open(file=filename, mode="r"):
        e = list(map(float, i.split()))
        ant.append(e[:-1]), con.append(e[-1])
    ant, con = np.array(ant), np.array(con)
    one, dlt = np.ptp(ant, axis=0) * 0.1, np.ptp(con) * 0.1
    les = np.linspace(np.min(con), np.max(con), 10)
    stm = np.min(ant, axis=0)
    return ant, one, con, les, dlt, stm


@cuda.reduce
def gpu_mul(a, b):
    return a * b


@cuda.reduce
def gpu_add(a, b):
    return a + b


threadshape, blockshape, attx, entx = (0, 0), (0, 0), 0, 0


@cuda.jit
def ActivateWeight(ant, r, one, wei, attn, entn):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    i = cuda.threadIdx.y + cuda.blockDim.y*cuda.blockIdx.x
    st = cuda.shared.array(threadshape, nb.float64)
    if x < attn and y < entn:
        st[x][y] = exp(-(ant[i][x]-r[x])/one[x])
    cuda.syncthreads()
    z = attn - 1
    while z != 0:
        if x <= z and x > z >> 1 and x < attn and y < entn:
            st[z >> 1][y] *= st[z][y]
        z >>= 1
        cuda.syncthreads()
    if x == 0:
        wei[i] = st[0][y]


@cuda.jit
def NormWeight(wei, sw):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    wei[x] /= sw


@cuda.jit
def ProbabilityMass(wei, con, lev, dlt, entn, mass):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if x < entn:
        ls = (lev - con[x]) / (0.25 * dlt) - 2.0
        rs = (lev - con[x]) / (0.25 * dlt) + 2.0
        b = 0.5 * (erf(rs/sqrt(2.0)) - erf(ls/sqrt(2.0)))
        mass[x] = (wei[x] * (b - 1.0) + 1.0) / (1.0 - wei[x])


def EvidentialReasonging(ant, con, r, one, dlt, wei, mas, les, attn, entn):
    ActivateWeight[blockshape, threadshape](ant, r, one, wei, attn, entn)
    sw = gpu_add(wei)
    NormWeight[int(entx/attx), 1024](wei, sw)
    B = np.empty(les.shape)
    for i in range(les.shape[0]):
        ProbabilityMass[int(entx/attx), 1024](wei, con, les[i], dlt, entn, mas)
        B[i] = gpu_mul(mas)
    S = np.sum(B)-les.shape[0]
    for i in range(les.shape[0]):
        B[i] = (B[i] - 1.0)/S
    return np.dot(les, B)


def main():
    ant, one, con, les, dlt, stm = ReadData("./data/oil_rev.txt")
    global threadshape, blockshape, attx, entx
    entn, attn = ant.shape
    attx, entx = 1, 1
    while attx < attn:
        attx <<= 1
    while entx * 1024 < entn * attx:
        entx <<= 1
    global threadshape, blockshape
    threadshape = (attx, int(1024 / attx))
    blockshape = (entx, 1)
    weight = cuda.device_array((entn), np.float64)
    mass = cuda.device_array((entn), np.float)
    for i in range(10):
        for j in range(10):
            r = np.array([stm[0] + i * one[0], stm[1] + j * one[1]])
            o = EvidentialReasonging(ant, con, r, one, dlt,
                                     weight, mass, les, attn, entn)
            with open("result", "a") as f:
                f.write(str(r[0])+" "+str(r[1])+" "+str(o)+"\n")


if __name__ == "__main__":
    main()
