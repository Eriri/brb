import numpy as np
from math import exp, sqrt, erf
from numba import cuda


def ReadData(filename):
    ant, con = [], []
    for i in open(file=filename, mode="r"):
        e = list(map(float, i.split()))
        ant.append(e[:-1]), con.append(e[-1])
    return np.array(ant), np.array(con)


@cuda.jit
def ActivateWeight(ant, r, one, wei, attn, entn, st):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    i = cuda.threadIdx.y + cuda.blockDim.y*cuda.blockIdx.x
    if x < attn and y < entn:
        st[y][x] = exp(-(ant[i][x]-r[x])/one[x])
    cuda.syncthreads()
    z = attn - 1
    while z != 0:
        if x <= z and x > z >> 1 and x < attn and y < entn:
            st[y][z >> 1] *= st[y][z]
        z >>= 1
        cuda.syncthreads()
    if x == 0:
        wei[i] = st[y][0]


@cuda.jit
def NormWeight(wei, sw):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    wei[x] /= sw


@cuda.jit
def ProbabilityMass(wei, con, lev, dlt, ent, mass):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if x < ent:
        ls = (lev - con[x]) / (0.25 * dlt) - 2.0
        rs = (lev - con[x]) / (0.25 * dlt) + 2.0
        mass[x] = 1.0 + wei[x] * 0.5 * \
            (erf(rs/sqrt(2.0)) - erf(ls/sqrt(2.0))) / (1.0 - wei[x])


def EvidentialReasonging(ant, con, r, one, wei, les, attn, entn, threadshape, blockshape, st):
    ActivateWeight[blockshape, threadshape](ant, r, one, wei, attn, entn, st)
    sw = gpu_add(wei)
    NormWeight(wei, sw)


@cuda.reduce
def gpu_mul(a, b):
    return a * b


@cuda.reduce
def gpu_add(a, b):
    return a + b


def main():
    ant, con = ReadData("oil_rev.txt")
    entn, attn = ant.shape
    attx, entx = 1, 1
    while attx < attn:
        attx <<= 1
    while entx * (1024 / attx) < entn:
        entx <<= 1
    threadshape = (attx, 1024 / attx)
    blockshape = (entx, 1)
    weight = cuda.arra
    st = cuda.shared.array(threadshape, np.float64)


if __name__ == "__main__":
    main()
