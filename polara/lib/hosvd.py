# python 2/3 interoperability
from __future__ import print_function
try:
    range = xrange
except NameError:
    pass

import numpy as np
from scipy.sparse.linalg import svds
from numba import njit


@njit(nogil=True)
def double_tensordot(idx, val, u, v, mode0, mode1, mode2, res):
    new_shape1 = u.shape[1]
    new_shape2 = v.shape[1]
    for i in range(len(val)):
        i0 = idx[i, mode0]
        i1 = idx[i, mode1]
        i2 = idx[i, mode2]
        vi = val[i]
        for j in range(new_shape1):
            for k in range(new_shape2):
                res[i0, j, k] += vi * u[i1, j] * v[i2, k]


def tensordot2(idx, val, shape, U, V, modes, dtype=None):
    mode1, mat_mode1 = modes[0]
    mode2, mat_mode2 = modes[1]

    u = U.T if mat_mode1 == 1 else U
    v = V.T if mat_mode2 == 1 else V

    mode0, = [x for x in (0, 1, 2) if x not in (mode1, mode2)]
    new_shape = (shape[mode0], U.shape[1 - mat_mode1], V.shape[1 - mat_mode2])

    res = np.zeros(new_shape, dtype=dtype)
    double_tensordot(idx, val, u, v, mode0, mode1, mode2, res)
    return res


def tucker_als(idx, val, shape, core_shape, iters=25, growth_tol=0.01, batch_run=False, seed=None):
    '''
    The function computes Tucker ALS decomposition of sparse tensor
    provided in COO format. Usage:
    u0, u1, u2, g = newtuck(idx, val, shape, core_shape)
    '''
    def log_status(msg):
        if not batch_run:
            print(msg)

    random_state = np.random if seed is None else np.random.RandomState(seed)

    r0, r1, r2 = core_shape
    u1 = random_state.rand(shape[1], r1)
    u1 = np.linalg.qr(u1, mode='reduced')[0]
    u2 = random_state.rand(shape[2], r2)
    u2 = np.linalg.qr(u2, mode='reduced')[0]

    g_norm_old = 0
    for i in range(iters):
        log_status('Step %i of %i' % (i + 1, iters))
        u0 = tensordot2(idx, val, shape, u2, u1, ((2, 0), (1, 0)))\
            .reshape(shape[0], r1 * r2)
        uu = svds(u0, k=r0, return_singular_vectors='u')[0]
        u0 = np.ascontiguousarray(uu[:, ::-1])

        u1 = tensordot2(idx, val, shape, u2, u0, ((2, 0), (0, 0)))\
            .reshape(shape[1], r0 * r2)
        uu = svds(u1, k=r1, return_singular_vectors='u')[0]
        u1 = np.ascontiguousarray(uu[:, ::-1])

        u2 = tensordot2(idx, val, shape, u1, u0, ((1, 0), (0, 0)))\
            .reshape(shape[2], r0 * r1)
        uu, ss, vv = svds(u2, k=r2)
        u2 = np.ascontiguousarray(uu[:, ::-1])

        g_norm_new = np.linalg.norm(ss)
        g_growth = (g_norm_new - g_norm_old) / g_norm_new
        g_norm_old = g_norm_new
        log_status('growth of the core: %f' % g_growth)
        if g_growth < growth_tol:
            log_status('Core is no longer growing. Norm of the core: %f' % g_norm_old)
            break

    g = np.ascontiguousarray((ss[:, np.newaxis] * vv)[::-1, :])
    g = g.reshape(r2, r1, r0).transpose(2, 1, 0)
    log_status('Done')
    return u0, u1, u2, g
