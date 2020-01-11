import tensorflow as tf


def activating_weight_categorical(x, a, rw, junc):
    '''
    [[None,cat_dim],...],[[rule_num,cat_dim],...],[rule_num]->[None,rule_num]
    a normalized, rw not negative
    '''
    w = [tf.reduce_sum(tf.math.abs(ai - tf.expand_dims(xi, -2)), -1) for xi, ai in zip(x, a)]
    w = [tf.reduce_sum(tf.math.sqrt(ai * tf.expand_dims(xi, -2)), -1) for xi, ai in zip(x, a)]
    if junc == 'con':
        return tf.reduce_prod(tf.concat([tf.expand_dims(replace_zero_with_one(wi), -1) for wi in w], -1), -1)
    if junc == 'dis':
        return tf.reduce_sum(tf.concat([tf.expand_dims(wi, -1) for wi in w], -1), -1)
    raise Exception('junc should be either con or dis')


def activating_weight_numerical(x, a, o, rw, junc):
    '''
    [None,att_dim],[rule_num,att_dim],[att_dim],[rule_num]->[None,rule_num]
    o and rw not negative
    '''
    w = tf.math.square((a - tf.expand_dims(x, -2))/o)
    if junc == 'con':
        return tf.math.exp(-tf.reduce_sum(replace_nan_with_zero(w), -1)) * rw
    if junc == 'dis':
        return tf.reduce_sum(replace_nan_with_zero(tf.math.exp(tf.negative(w))), -1) * rw
    raise Exception('junc should be either con or dis')


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
        st[x][y] = exp(-abs(ant[i][x]-r[x])/one[x])
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


@jit
def EvidentialReasonging(ant, con, r, one, dlt, wei, mas, les, attn, entn):
    ActivateWeight[blockshape, threadshape](ant, r, one, wei, attn, entn)
    sw = gpu_add(wei)
    NormWeight[int(entx/attx), 1024](wei, sw)
    B = np.empty(les.shape)
    for i in range(0, les.shape[0]):
        ProbabilityMass[int(entx/attx), 1024](wei, con, les[i], dlt, entn, mas)
        B[i] = gpu_mul(mas, init=1.0)
    S = np.sum(B)-les.shape[0]
    for i in range(0, les.shape[0]):
        B[i] = (B[i] - 1.0)/(S + 1e-6)
    return np.dot(les, B)


def EvidentialReasoing(a, ant, con, rw):
    w = np.exp(-0.5 * np.sum((ant-a)*(ant-a)/(one*one), axis=1)) * rw
    sw, b = np.sum(w), np.random.uniform(size=len(les))
    if np.max(w) == sw:
        return [b/np.sum(b), con[np.argmax(w)]][int(sw != 0.0)]
    b = np.prod(np.transpose(con) * w / (sw - w) + 1, axis=1) - 1
    return b / np.sum(b)


def GradientDescent(ta, tc, bs, mua, muc, muw, ba, bc, bw):
    bi = np.arange(len(ta)-len(ta) % bs)
    np.random.shuffle(bi)
    pb = tqdm.tqdm(total=len(bi)/bs)
    for i, bm in enumerate(np.split(bi, len(bi)/bs)):
        t = i % len(ba)
        nx, ny, nz = np.zeros(ba[t].shape), np.zeros(bc[t].shape), 0.0  # (T),(N),()
        for a, c in zip(ta[bm], tc[bm]):
            theta = 1 / (1 + np.exp(-bw))  # (L)
            alpha = np.exp(-np.sum((ba-a)*(ba-a)/(one*one), axis=1) / 2)  # (L)
            w = alpha * theta  # (L)
            sw = np.sum(w)
            if np.max(w) == sw:
                continue
            B = np.transpose(np.exp(bc) / np.sum(np.exp(bc), axis=1)[:, None])  # (N,L)
            Bi = B * w / (sw - w) + 1  # (N,L)
            B_ = np.prod(Bi, axis=1) - 1  # (N)
            pc = B_ / np.sum(B_)  # (N)
            dBB_ = (np.sum(B_) - B_) / (np.sum(B_) * np.sum(B_))  # (N)

            bn1 = - theta[t] * B * w / (sw - w) / (sw - w) * (np.prod(Bi, axis=1)[:, None] / Bi)  # (N,L)
            bn2 = theta[t] * B[:, t] / (sw - w[t]) * np.prod(Bi, axis=1) / Bi[:, t]  # (N)
            dB_a = np.sum(bn1, axis=1) - bn1[:, t] + bn2  # (N)
            dax = (a - ba[t]) / one / one * np.exp(-np.sum((a - ba[t])*(a - ba[t]) / one / one) / 2)  # (T)
            nx += np.sum((pc - c) * dBB_ * dB_a) * dax

            dB_b = w[t] / (sw - w[t]) * np.prod(Bi, axis=1) / Bi[:, t]  # (N)
            dby = np.exp(bc[t]) * (np.sum(np.exp(bc[t])) - np.exp(bc[t])) / (np.sum(np.exp(bc[t])) * np.sum(np.exp(bc[t])))  # (N)
            ny += (pc - c) * dBB_ * dB_b * dby

            bn3 = - alpha[t] * B * w / ((sw - w) * (sw - w)) * (np.prod(Bi, axis=1)[:, None] / Bi)  # (N,L)
            bn4 = alpha[t] * B[:, t] / (sw - w[t]) * np.prod(Bi, axis=1) / Bi[:, t]  # (N)
            dB_t = np.sum(bn3, axis=1) - bn3[:, t] + bn4  # (N)
            dtz = theta[t] * (1.0 - theta[t])
            nz += np.sum((pc - c) * dBB_ * dB_t * dtz)

        ba[t] -= mua * nx / bs
        bc[t] -= muc * ny / bs
        bw[t] -= muw * nz / bs
        pb.update()
    pb.close()
    return ba, bc, bw
