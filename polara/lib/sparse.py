import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import numpy as np
from numpy import power
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import norm as spnorm

from numba import jit, njit, guvectorize, prange
from numba import float64 as f8
from numba import intp as ip

from polara.recommender import defaults


tuplsize = sys.getsizeof(())
itemsize = np.dtype(np.intp).itemsize
pntrsize = sys.getsizeof(1.0)
# size of list of tuples of indices - to estimate when to convert sparse matrix to dense
# based on http://stackoverflow.com/questions/15641344/python-memory-consumption-dict-vs-list-of-tuples
# and https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
def get_nnz_max():
    return int(defaults.memory_hard_limit * (1024**3) / (tuplsize + 2*(pntrsize + itemsize)))


def check_sparsity(matrix, nnz_coef=0.5, tocsr=False):
    if matrix.nnz > nnz_coef * matrix.shape[0] * matrix.shape[1]:
        return matrix.toarray(order='C')
    if tocsr:
        return matrix.tocsr()
    return matrix


def sparse_dot(left_mat, right_mat, dense_output=False, tocsr=False):
    # scipy always returns sparse result, even if dot product is dense
    # this function offers solution to this problem
    # it also takes care on sparse result w.r.t. to further processing
    if dense_output:  # calculate dense result directly
        # TODO matmat multiplication instead of iteration with matvec
        res_type = np.result_type(right_mat.dtype, left_mat.dtype)
        result = np.empty((left_mat.shape[0], right_mat.shape[1]), dtype=res_type)
        for i in range(left_mat.shape[0]):
            v = left_mat.getrow(i)
            result[i, :] = csc_matvec(right_mat, v, dense_output=True, dtype=res_type)
    else:
        result = left_mat.dot(right_mat.T)
        # NOTE even though not neccessary for symmetric i2i matrix,
        # transpose helps to avoid expensive conversion to CSR (performed by scipy)
        if result.nnz > get_nnz_max():
            # too many nnz lead to undesired memory overhead in downvote_seen_items
            result = result.toarray() # not using order='C' as it may consume memory
        else:
            result = check_sparsity(result, tocsr=tocsr)
    return result


def rescale_matrix(matrix, scaling, axis, binary=True):
    '''Function to scale either rows or columns of the sparse rating matrix'''
    if scaling == 1: # no scaling (standard SVD case)
        return matrix

    if binary:
        norm = np.sqrt(matrix.getnnz(axis=axis)) # compute Euclidean norm as if values are binary
    else:
        norm = spnorm(matrix, axis=axis, ord=2) # compute Euclidean norm

    scaling_matrix = diags(power(norm, scaling-1, where=norm!=0))

    if axis == 0: # scale columns
        return matrix.dot(scaling_matrix)
    if axis == 1: # scale rows
        return scaling_matrix.dot(matrix)


# matvec implementation is based on
# http://stackoverflow.com/questions/18595981/improving-performance-of-multiplication-of-scipy-sparse-matrices
@njit(nogil=True)
def matvec2dense(m_ptr, m_ind, m_val, v_nnz, v_val, out):
    l = len(v_nnz)
    for j in range(l):
        col_start = v_nnz[j]
        col_end = col_start + 1
        ind_start = m_ptr[col_start]
        ind_end = m_ptr[col_end]
        if ind_start != ind_end:
            out[m_ind[ind_start:ind_end]] += m_val[ind_start:ind_end] * v_val[j]


@njit(nogil=True)
def matvec2sparse(m_ptr, m_ind, m_val, v_nnz, v_val, sizes, indices, data):
    l = len(sizes) - 1
    for j in range(l):
        col_start = v_nnz[j]
        col_end = col_start + 1
        ind_start = m_ptr[col_start]
        ind_end = m_ptr[col_end]
        data_start = sizes[j]
        data_end = sizes[j+1]
        if ind_start != ind_end:
            indices[data_start:data_end] = m_ind[ind_start:ind_end]
            data[data_start:data_end] = m_val[ind_start:ind_end] * v_val[j]


def csc_matvec(mat_csc, vec, dense_output=True, dtype=None):
    v_nnz = vec.indices
    v_val = vec.data

    m_val = mat_csc.data
    m_ind = mat_csc.indices
    m_ptr = mat_csc.indptr

    res_dtype = dtype or np.result_type(mat_csc.dtype, vec.dtype)
    if dense_output:
        res = np.zeros((mat_csc.shape[0],), dtype=res_dtype)
        matvec2dense(m_ptr, m_ind, m_val, v_nnz, v_val, res)
    else:
        sizes = m_ptr.take(v_nnz+1) - m_ptr.take(v_nnz)
        sizes = np.concatenate(([0], np.cumsum(sizes)))
        n = sizes[-1]
        data = np.empty((n,), dtype=res_dtype)
        indices = np.empty((n,), dtype=np.intp)
        indptr = np.array([0, n], dtype=np.intp)
        matvec2sparse(m_ptr, m_ind, m_val, v_nnz, v_val, sizes, indices, data)
        res = csr_matrix((data, indices, indptr), shape=(1, mat_csc.shape[0]), dtype=res_dtype)
        res.sum_duplicates() # expensive operation
    return res


@njit
def _blockify(ind, ptr, major_dim):
    # convenient function to compute only diagonal
    # elements of the product of 2 matrices;
    # indices must be intp in order to avoid overflow
    # major_dim is shape[0] for csc format and shape[1] for csr format
    n = len(ptr) - 1
    for i in range(1, n): #first row/col is unchanged
        lind = ptr[i]
        rind = ptr[i+1]
        for j in range(lind, rind):
            shift_ind = i * major_dim
            ind[j] += shift_ind


def row_unblockify(mat, block_size):
    # only for CSR matrices
    factor = (mat.indices // block_size) * block_size
    mat.indices -= factor
    mat._shape = (mat.shape[0], block_size)


def row_blockify(mat, block_size):
    # only for CSR matrices
    _blockify(mat.indices, mat.indptr, block_size)
    mat._shape = (mat.shape[0], block_size*mat.shape[0])


def inverse_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def unfold_tensor_coordinates(index, shape, mode):
    # TODO implement direct calculation w/o intermediate flattening
    modes = [m for m in [0, 1, 2] if m != mode] + [mode,]
    mode_shape = tuple(shape[m] for m in modes)
    mode_index = tuple(index[m] for m in modes)
    flat_index = np.ravel_multi_index(mode_index, mode_shape)

    unfold_shape = (mode_shape[0]*mode_shape[1], mode_shape[2])
    unfold_index = np.unravel_index(flat_index, unfold_shape)
    return unfold_index, unfold_shape


def tensor_outer_at(vtarget, **kwargs):
    @guvectorize(['void(float64[:], float64[:, :], float64[:, :], intp[:], intp[:], float64[:, :])'],
                 '(),(i,m),(j,n),(),()->(m,n)',
                 target=vtarget, nopython=True, **kwargs)
    def tensor_outer_wrapped(val, v, w, i, j, res):
        r1 = v.shape[1]
        r2 = w.shape[1]
        for m in range(r1):
            for n in range(r2):
                res[m, n] = val[0] * v[i[0], m] * w[j[0], n]
    return tensor_outer_wrapped


@njit(nogil=True)
def dttm_seq(idx, val, u, v, mode0, mode1, mode2, res):
    new_shape1 = u.shape[1]
    new_shape2 = v.shape[1]
    for i in range(len(val)):
        i0 = idx[i, mode0]
        i1 = idx[i, mode1]
        i2 = idx[i, mode2]
        vv = val[i]
        for j in range(new_shape1):
            uij = u[i1, j]
            for k in range(new_shape2):
                vik = v[i2, k]
                res[i0, j, k] += vv * uij * vik


@njit(parallel=True)
def dttm_par(idx, val, mat1, mat2, mode1, mode2, unqs, inds, res):
    r1 = mat1.shape[1]
    r2 = mat2.shape[1]
    n = len(unqs)

    for s in prange(n):
        i0 = unqs[s]
        ul = inds[s]
        for pos in ul:
            i1 = idx[pos, mode1]
            i2 = idx[pos, mode2]
            vp = val[pos]
            for j1 in range(r1):
                for j2 in range(r2):
                    res[i0, j1, j2] += vp * mat1[i1, j1] * mat2[i2, j2]


# @jit(parallel=True) # numba up to v0.41.dev only supports the 1st argument
# https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
def arrange_index(array):
    unqs, unq_inv, unq_cnt = np.unique(array, return_inverse=True, return_counts=True)
    inds = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    return unqs, inds

def arrange_indices(idx, mode_mask=None):
    n = idx.shape[1]
    res = [[]]*n
    if mode_mask is None:
        mode_mask = [True] * n

    if sum(mode_mask) == 0:
        return res

    if sum(mode_mask) == 1:
        mode, = [i for i, x in enumerate(mode_mask) if x]
        res[mode] = arrange_index(idx[:, mode])
        return res

    with ThreadPoolExecutor(max_workers=sum(mode_mask)) as executor:
        arranged_futures = {executor.submit(arrange_index, idx[:, mode]): mode
                            for mode in range(n) if mode_mask[mode]}
        for future in as_completed(arranged_futures):
            mode = arranged_futures[future]
            res[mode] = future.result()
    return res
