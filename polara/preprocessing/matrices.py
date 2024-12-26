import numpy as np
from numpy import power
from scipy.sparse import diags
from scipy.sparse.linalg import norm as spnorm
from numba import njit
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


def find_top_influential_items(matrix, user_coverage=0.8):
    '''
    Find minimal a set of items at least `user_coverage` unique users have interacted with.

    This is a slightly faster (~2x speedup on ML-1M) implementation of a greedy approach:
    ```python
    total_users, total_items = matrix.shape
    n_users = int(user_coverage * total_users) if user_coverage < 1 else user_coverage
    covered_users = np.zeros(total_items, dtype=bool)
    item_set = [] # to collect found items
    while np.count_nonzero(covered_users) < n_users:
        top_item = mode(matrix[~covered_users].indices, total_items)
        item_set.append(top_item)
        covered_users += np.logical_or.reduceat(matrix.indices==top_item, matrix.indptr[:-1])
    ```

    Note
    ========
    This is an instance of the so called "set cover problem", which is generally NP-complete.
    It is known that the greedy approach provides the best approximation in polynonial time.
    See https://en.wikipedia.org/wiki/Set_cover_problem for more details.

    One of the most straightforward (and slow) implementations reads:
    ```python
    def set_cover(matrix):
        indices = matrix.indices
        indptr = matrix.indptr
        universe = np.arange(matrix.shape[1])
        cover = []
        while len(universe):
            subset_sizes = np.add.reduceat(np.in1d(indices, universe), indptr[:-1])
            idx = np.argmax(subset_sizes)
            cover.append(idx)
            universe = np.setdiff1d(
                universe,
                indices[indptr[idx]:indptr[idx+1]],
                assume_unique=True
            )
        return cover
    ```
    '''
    assert matrix.has_canonical_format # we rely on sorted indices and no duplicates
    assert matrix.format == 'csr'

    total_users, total_items = matrix.shape
    n_users = int(user_coverage * total_users) if user_coverage < 1 else min(user_coverage, total_users)
    useridx = np.arange(total_users)

    n_covered = 0 # to track the number of unique users found
    item_set = [] # to collect found items
    matrix_rem = matrix # will shrink this matrix by excluding found users from consideration
    for _ in range(total_items): # normally would terminate before exhausting full range
        top_item = find_most_common_element(matrix_rem.indices, total_items) # TODO: reuse previous counts
        item_set.append(top_item)
        top_inds = np.flatnonzero(matrix_rem.indices == top_item)
        users = np.searchsorted(matrix_rem.indptr, top_inds, 'right') - 1
        n_covered += len(users)
        if n_covered >= n_users:
            break
        remaining_users = np.isin(useridx[:matrix_rem.shape[0]], users, assume_unique=True, invert=True)
        matrix_rem = matrix_rem[remaining_users]
    return item_set


@njit
def find_most_common_element(indices, n_elements):
    counter = np.zeros(n_elements, dtype=np.intp)
    top_freq, top_element = 0, -1
    for element in indices:
        counter[element] = element_freq = counter[element] + 1
        if element_freq > top_freq:
            top_freq, top_element = element_freq, element
    # TODO: there's no need to parse entire array
    # after certain point, the frequency of most common element so far
    # becomes unreachable, we can use this fact to exit the loop early
    # however, will need to track second most frequent element
    # the difference in frequencies of most common and second most common
    # elements will be used as an indicator for early stopping
    return top_element