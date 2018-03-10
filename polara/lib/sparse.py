# python 2/3 interoperability
try:
    range = xrange
except NameError:
    pass

import numpy as np
from scipy.sparse import csr_matrix
from numba import jit

# matvec implementation is based on
# http://stackoverflow.com/questions/18595981/improving-performance-of-multiplication-of-scipy-sparse-matrices


@jit(nopython=True, nogil=True)
def matvec2dense(m_ptr, m_ind, m_val, v_nnz, v_val, out):
    l = len(v_nnz)
    for j in range(l):
        col_start = v_nnz[j]
        col_end = col_start + 1
        ind_start = m_ptr[col_start]
        ind_end = m_ptr[col_end]
        if ind_start != ind_end:
            out[m_ind[ind_start:ind_end]] += m_val[ind_start:ind_end] * v_val[j]


@jit(nopython=True, nogil=True)
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


@jit(nopython=True)
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
