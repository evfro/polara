import numpy as np
from numba import jit

@jit(nopython=True, nogil=True)
def double_tensordot(idx, val, U, V, new_shape1, new_shape2, ten_mode0, ten_mode1, ten_mode2, res):
    I = idx.shape[0]
    J = new_shape1
    K = new_shape2
    for i in xrange(I):
        i0 = idx[i, ten_mode0]
        i1 = idx[i, ten_mode1]
        i2 = idx[i, ten_mode2]
        for j in xrange(J):
            for k in xrange(K):
                res[i0, j, k] += val[i] * U[i1, j] * V[i2, k]


def tensordot2(idx, val, shape, U, V, modes):
    ten_mode1, mat_mode1 = modes[0]
    ten_mode2, mat_mode2 = modes[1]

    ten_mode0, = [x for x in (0,1,2) if x not in (ten_mode1, ten_mode2)]
    new_shape = (shape[ten_mode0], U.shape[1-mat_mode1], V.shape[1-mat_mode2])
    res = np.zeros(new_shape)

    if mat_mode1==1:
        vU = U.T
    else:
        vU = U

    if mat_mode2==1:
        vV = V.T
    else:
        vV = V

    double_tensordot(idx, val, vU, vV, new_shape[1], new_shape[2], ten_mode0, ten_mode1, ten_mode2, res)
    return res

def tucker_als(idx, val, shape, core_shape, iters=25, growth_tol=0.01, batch_run=False):
    '''
    The function computes Tucker ALS decomposition of sparse tensor
    provided in COO format. Usage:
    u0, u1, u2, g = newtuck(idx, val, shape, core_shape)
    '''
    def log_status(msg):
        if not batch_run:
            print msg

    if not (idx.flags.c_contiguous and val.flags.c_contiguous):
        raise ValueError('Warning! Imput arrays must be C-contigous.')


    #TODO: it's better to implement check for future
    #if np.any(idx[1:, 0]-idx[:-1, 0]) < 0):
    #    print 'Warning! Index array must be sorted by first column in ascending order.'

    r0, r1, r2 = core_shape

    u1 = np.random.rand(shape[1], r1)
    u1 = np.linalg.qr(u1, mode='reduced')[0]

    u2 = np.random.rand(shape[2], r2)
    u2 = np.linalg.qr(u2, mode='reduced')[0]

    u1 = np.ascontiguousarray(u1)
    u2 = np.ascontiguousarray(u2)

    g_norm_old = 0

    for i in xrange(iters):
        log_status('Step %i of %i' % (i+1, iters))
        u0 = tensordot2(idx, val, shape, u2, u1, ((2, 0), (1, 0)))\
            .reshape(shape[0], r1*r2)
        uu = np.linalg.svd(u0, full_matrices=0)[0]
        u0 = np.ascontiguousarray(uu[:, :r0])

        u1 = tensordot2(idx, val, shape, u2, u0, ((2, 0), (0, 0)))\
            .reshape(shape[1], r0*r2)
        uu = np.linalg.svd(u1, full_matrices=0)[0]
        u1 = np.ascontiguousarray(uu[:, :r1])

        u2 = tensordot2(idx, val, shape, u1, u0, ((1, 0), (0, 0)))\
            .reshape(shape[2], r0*r1)
        uu, ss, vv = np.linalg.svd(u2, full_matrices=0)
        u2 = np.ascontiguousarray(uu[:, :r2])

        g_norm_new = np.linalg.norm(np.diag(ss[:r2]))
        g_growth = (g_norm_new - g_norm_old) / g_norm_new
        g_norm_old = g_norm_new
        log_status('growth of the core: %f' % g_growth)
        if g_growth < growth_tol:
            log_status('Core is no longer growing. Norm of the core: %f' % g_norm_old)
            break

    g = np.diag(ss[:r2]).dot(vv[:r2, :])
    g = g.reshape(r2, r1, r0).transpose(2, 1, 0)
    log_status('Done')
    return u0, u1, u2, g
