import numpy as np
from scipy.sparse.linalg import svds

from polara.lib.sparse import arrange_indices
from polara.lib.sparse import dttm_seq, dttm_par


def ttm3d_seq(idx, val, shape, U, V, modes, dtype=None):
    mode1, mat_mode1 = modes[0]
    mode2, mat_mode2 = modes[1]

    u = U.T if mat_mode1 == 1 else U
    v = V.T if mat_mode2 == 1 else V

    mode0, = [x for x in (0, 1, 2) if x not in (mode1, mode2)]
    new_shape = (shape[mode0], U.shape[1 - mat_mode1], V.shape[1 - mat_mode2])

    res = np.zeros(new_shape, dtype=dtype)
    dttm_seq(idx, val, u, v, mode0, mode1, mode2, res)
    return res


def ttm3d_par(idx, val, shape, U, V, modes, unqs, inds, dtype=None):
    mode1, mat_mode1 = modes[0]
    mode2, mat_mode2 = modes[1]

    u = U.T if mat_mode1 == 1 else U
    v = V.T if mat_mode2 == 1 else V

    mode0, = [x for x in (0, 1, 2) if x not in (mode1, mode2)]
    new_shape = (shape[mode0], U.shape[1 - mat_mode1], V.shape[1 - mat_mode2])

    res = np.zeros(new_shape, dtype=dtype)
    dttm_par(idx, val, u, v, mode1, mode2, unqs, inds, res)
    return res


def hooi(idx, val, shape, core_shape, num_iters=25, parallel_ttm=False, growth_tol=0.01, verbose=False, seed=None):
    '''
    Compute Tucker decomposition of a sparse tensor in COO format
    with the help of HOOI algorithm. Usage:
    u0, u1, u2, g = hooi(idx, val, shape, core_shape)
    '''
    def log_status(msg):
        if verbose:
            print(msg)

    tensor_data = idx, val, shape
    if not isinstance(parallel_ttm, (list, tuple)):
        parallel_ttm = [parallel_ttm] * len(shape)

    assert len(shape) == len(parallel_ttm)

    index_data = arrange_indices(idx, parallel_ttm)
    ttm = [ttm3d_par if par else ttm3d_seq for par in parallel_ttm]

    random_state = np.random if seed is None else np.random.RandomState(seed)

    r0, r1, r2 = core_shape
    u1 = random_state.rand(shape[1], r1)
    u1 = np.linalg.qr(u1, mode='reduced')[0]
    u2 = random_state.rand(shape[2], r2)
    u2 = np.linalg.qr(u2, mode='reduced')[0]

    g_norm_old = 0
    for i in range(num_iters):
        log_status('Step %i of %i' % (i + 1, num_iters))

        u0 = ttm[0](*tensor_data, u2, u1, ((2, 0), (1, 0)), *index_data[0]).reshape(shape[0], r1 * r2)
        uu = svds(u0, k=r0, return_singular_vectors='u')[0]
        u0 = np.ascontiguousarray(uu[:, ::-1])

        u1 = ttm[1](*tensor_data, u2, u0, ((2, 0), (0, 0)), *index_data[1]).reshape(shape[1], r0 * r2)
        uu = svds(u1, k=r1, return_singular_vectors='u')[0]
        u1 = np.ascontiguousarray(uu[:, ::-1])

        u2 = ttm[2](*tensor_data, u1, u0, ((1, 0), (0, 0)), *index_data[2]).reshape(shape[2], r0 * r1)
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
