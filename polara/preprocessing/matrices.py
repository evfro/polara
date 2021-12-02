import numpy as np
from numpy import power
from scipy.sparse import diags
from scipy.sparse.linalg import norm as spnorm
import pandas as pd
from polara.tools.random import check_random_state


def split_holdout(matrix, sample_max_rated=True, random_state=None):
    '''
    Uses CSR format to efficiently access non-zero elements.
    '''
    holdout = []
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data

    random_state = check_random_state(random_state)
    for i in range(len(indptr)-1): # for every user i
        head = indptr[i]
        tail = indptr[i+1]
        candidates = indices[head:tail]
        if sample_max_rated:
            vals = data[head:tail] # user feedback
            pos_max, = np.where(vals == vals.max())
            candidates = candidates[pos_max]
        holdout.append(random_state.choice(candidates))
    return np.array(holdout)


def mask_holdout(matrix, holdout_items, copy=True):
    '''
    Zeroize holdout items in the rating matrix.
    Requires exactly 1 holdout item per each row.
    '''
    masked = matrix.copy() if copy else matrix
    masked[np.arange(len(holdout_items)), holdout_items] = 0
    masked.eliminate_zeros()
    return masked


def sample_unseen_interactions(observations, holdout_items, size=999, random_state=None):
    '''
    Randomly samples unseen items per every user from observations matrix.
    Takes into account holdout items. Assumes there's only one holdout item per every user.
    '''
    n_users, n_items = observations.shape
    indptr = observations.indptr
    indices = observations.indices

    sample = np.zeros((n_users, size), dtype=indices.dtype)
    random_state = check_random_state(random_state)
    for i in range(len(indptr)-1):
        head = indptr[i]
        tail = indptr[i+1]
        seen_items = np.concatenate(([holdout_items[i]], indices[head:tail]))
        rand_items = sample_unseen(n_items, size, seen_items, random_state)
        sample[i, :] = rand_items
    return sample


def sample_unseen(pool_size, sample_size, exclude, random_state=None):
    '''Efficient sampling from a range with exclusion.'''
    assert (pool_size-len(exclude)) >= sample_size
    random_state = check_random_state(random_state)
    src = random_state.rand(pool_size)
    np.put(src, exclude, -1) # will never get to the top
    return np.argpartition(src, -sample_size)[-sample_size:]


def rescale_matrix(matrix, scaling, axis, binary=True, return_scaling_values=False):
    '''Function to scale either rows or columns of the sparse rating matrix'''
    result = None
    scaling_values = None
    if scaling == 1: # no scaling (standard SVD case)
        result = matrix

    if binary:
        norm = np.sqrt(matrix.getnnz(axis=axis)) # compute Euclidean norm as if values are binary
    else:
        norm = spnorm(matrix, axis=axis, ord=2) # compute Euclidean norm

    scaling_values = power(norm, scaling-1, where=norm != 0)
    scaling_matrix = diags(scaling_values)

    if axis == 0: # scale columns
        result = matrix.dot(scaling_matrix)
    if axis == 1: # scale rows
        result = scaling_matrix.dot(matrix)

    if return_scaling_values:
        result = (result, scaling_values)
    return result


def generate_banded_form(matrix):
    matrix = matrix.todia()
    bands = matrix.data
    offsets = matrix.offsets
    num_l = (offsets < 0).sum()
    num_u = (offsets > 0).sum()
    return (num_l, num_u), bands[np.argsort(offsets)[::-1], :]