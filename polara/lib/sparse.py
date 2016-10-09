import numpy as np
import scipy as sp
from scipy import sparse
from numba import jit

# matvec implementation is based on
# http://stackoverflow.com/questions/18595981/improving-performance-of-multiplication-of-scipy-sparse-matrices

@jit(nopython=True, nogil=True)
def matvec2dense(m_ptr, m_ind, m_val, v_nnz, v_val, out):
    l = len(v_nnz)
    for j in xrange(l):
        col_start = v_nnz[j]
        col_end = col_start + 1
        ind_start = m_ptr[col_start]
        ind_end = m_ptr[col_end]
        if ind_start != ind_end:
            out[m_ind[ind_start:ind_end]] += m_val[ind_start:ind_end] * v_val[j]


@jit(nopython=True, nogil=True)
def matvec2sparse(m_ptr, m_ind, m_val, v_nnz, v_val, sizes, indices, data):
    l = len(sizes) - 1
    for j in xrange(l):
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
        res = sp.sparse.csr_matrix((data, indices, indptr), shape=(1, mat_csc.shape[0]), dtype=res_dtype)
        res.sum_duplicates() # expensive operation
    return res
