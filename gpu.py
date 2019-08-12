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
def ActivateWeight(ant, r, one, wei, att, ent, st):
    x = cuda.threadIdx.x
    y = cuda.threadIdx.y
    i = cuda.threadIdx.y + cuda.blockDim.y*cuda.blockIdx.x
    if x < att and y < ent:
        st[y][x] = exp(-(ant[i][x]-r[x])/one[x])
    cuda.syncthreads()
    z = att - 1
    while z != 0:
        if x <= z and x > z >> 1 and x < att and y < ent:
            st[y][z >> 1] *= st[y][z]
        z >>= 1
        cuda.syncthreads()
    if x == 0:
        wei[i] = st[y][0]


@cuda.jit
def NormWeight(wei, sum):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    wei[x] /= sum


@cuda.jit
def ProbabilityMass(wei, con, lev, dlt, ent, mass):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if x < ent:
        ls = (lev - con[x]) / (0.25 * dlt) - 2.0
        rs = (lev - con[x]) / (0.25 * dlt) + 2.0
        mass[x] = 1.0 + wei[x] * 0.5 * \
            (erf(rs/sqrt(2.0)) - erf(ls/sqrt(2.0))) / (1.0 - wei[x])


def EvidentialReasonging():


@cuda.reduce
def gpu_prod(a, b):
    return a * b


@cuda.reduce
def gpu_sum(a, b):
    return a + b


def main():
    ant, con = ReadData("oil_rev.txt")


if __name__ == "__main__":
    main()
