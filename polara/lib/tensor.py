import numpy as np
from scipy.sparse.linalg import svds
from polara.lib.sparse import dttm_seq, dttm_par
from polara.lib.sparse import arrange_indices
from polara.tools.display import log_status


def core_growth_callback(growth_tol, verbose=True):
    def check_core_growth(step, core, factors):
        singular_values, _ = core
        core_norm = np.linalg.norm(singular_values)
        g_growth = (core_norm - check_core_growth.core_norm) / core_norm
        check_core_growth.core_norm = core_norm
        log_status(f'growth of the core: {g_growth}')
        if g_growth < growth_tol:
            log_status(f'Core is no longer growing. Norm of the core: {core_norm}.', verbose=verbose)
            raise StopIteration
    check_core_growth.core_norm = 0
    return check_core_growth


def ttm3d_seq(idx, val, shape, U, V, modes, dtype=None):
    mode1, mat_mode1 = modes[0]
    mode2, mat_mode2 = modes[1]

    u = U.T if mat_mode1 == 1 else U
    v = V.T if mat_mode2 == 1 else V

    mode0, = [x for x in (0, 1, 2) if x not in (mode1, mode2)]
    new_shape = (shape[mode0], U.shape[1-mat_mode1], V.shape[1-mat_mode2])

    res = np.zeros(new_shape, dtype=dtype)
    dttm_seq(idx, val, u, v, mode0, mode1, mode2, res)
    return res


def ttm3d_par(idx, val, shape, U, V, modes, unqs, inds, dtype=None):
    mode1, mat_mode1 = modes[0]
    mode2, mat_mode2 = modes[1]

    u = U.T if mat_mode1 == 1 else U
    v = V.T if mat_mode2 == 1 else V

    mode0, = [x for x in (0, 1, 2) if x not in (mode1, mode2)]
    new_shape = (shape[mode0], U.shape[1-mat_mode1], V.shape[1-mat_mode2])

    res = np.zeros(new_shape, dtype=dtype)
    dttm_par(idx, val, u, v, mode1, mode2, unqs, inds, res)
    return res


def initialize_factors(dims, ranks, seed):
    random_state = np.random if seed is None else np.random.RandomState(seed)
    factors = []
    for dim, rank in zip(dims, ranks):
        u_rnd = random_state.rand(dim, rank)
        u = np.linalg.qr(u_rnd, mode='reduced')[0]
        factors.append(u)
    return factors


def hooi(idx, val, shape, core_shape, return_core=True, num_iters=10,
         parallel_ttm=False, growth_tol=0.001, iter_callback=None,
         verbose=False, seed=None):
    '''
    Compute Tucker decomposition of a sparse tensor in COO format
    with the help of HOOI algorithm. Usage:
    u0, u1, u2, g = hooi(idx, val, shape, core_shape)
    '''
    tensor_data = idx, val, shape
    if not isinstance(parallel_ttm, (list, tuple)):
        parallel_ttm = [parallel_ttm] * len(shape)

    assert len(shape) == len(parallel_ttm)

    index_data = arrange_indices(idx, parallel_ttm)
    ttm = [ttm3d_par if par else ttm3d_seq for par in parallel_ttm]

    if iter_callback is None:
        iter_callback = core_growth_callback(growth_tol, verbose=verbose)
    iter_callback.stop_reason = 'Exceeded max iterations limit.'

    u1, u2 = initialize_factors(shape[1:], core_shape[1:], seed)
    g = None
    r0, r1, r2 = core_shape
    return_core_vectors = True if return_core else 'u'
    for i in range(num_iters):
        log_status('Step %i of %i' % (i+1, num_iters), verbose=verbose)

        u0 = ttm[0](*tensor_data, u2, u1, ((2, 0), (1, 0)), *index_data[0]).reshape(shape[0], r1*r2)
        uu, *_ = svds(u0, k=r0, return_singular_vectors='u')
        u0 = np.ascontiguousarray(uu[:, ::-1])

        u1 = ttm[1](*tensor_data, u2, u0, ((2, 0), (0, 0)), *index_data[1]).reshape(shape[1], r0*r2)
        uu, *_ = svds(u1, k=r1, return_singular_vectors='u')
        u1 = np.ascontiguousarray(uu[:, ::-1])

        u2 = ttm[2](*tensor_data, u1, u0, ((1, 0), (0, 0)), *index_data[2]).reshape(shape[2], r0*r1)
        uu, *core = svds(u2, k=r2, return_singular_vectors=return_core_vectors)
        u2 = np.ascontiguousarray(uu[:, ::-1])

        try:
            iter_callback(i, core, (u0, u1, u2))
        except StopIteration:
            iter_callback.stop_reason = 'Stopping criteria met.'
            break

    if return_core:
        ss, vv = core
        g = (
            np.ascontiguousarray((ss[:, np.newaxis] * vv)[::-1, :])
            .reshape(r2, r1, r0)
            .transpose(2, 1, 0)
        )
    log_status('Done')
    return u0, u1, u2, g
