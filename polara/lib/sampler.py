import heapq
from random import randrange
from random import seed as set_seed
import numpy as np
from numba import njit, prange

# Sampler below is an adapted "jit-friendly" version of the accepted answer at
# https://stackoverflow.com/questions/18921302/how-to-incrementally-sample-without-replacement


@njit(fastmath=True)
def prime_sampler_state(n, exclude):
    """
    Initialize state to be used in fast sampler. Helps ensure excluded items are
    never sampled by placing them outside of sampling region.
    """
    # initialize typed numba dicts
    state = {n: n}
    state.pop(n)
    track = {n: n}
    track.pop(n)

    n_pos = n - len(state) - 1
    # reindex excluded items, placing them in the end
    for i, item in enumerate(exclude):
        pos = n_pos - i
        x = track.get(item, item)
        t = state.get(pos, pos)
        state[x] = t
        track[t] = x
        state.pop(pos, n)
        track.pop(item, n)
    return state


@njit(fastmath=True)
def sample_unseen(n, size, exclude):
    """
    This is a generator to sample a desired number of integers from a range
    (starting from zero) excluding black-listed elements. It samples items
    one by one, which makes it memory efficient and convenient for "on-the-fly"
    calculations with sampled elements.
    """
    # exclude items by moving them out of samling region
    state = prime_sampler_state(n, exclude)
    remaining = n - len(exclude)
    # gradually sample from the decreased size range
    for _ in range(size):
        i = randrange(remaining)
        yield state[i] if i in state else i  # avoid numba bug with dict.get(i,i)
        remaining -= 1
        state[i] = state.get(remaining, remaining)
        state.pop(remaining, n)


@njit(fastmath=True)
def sample_fill(sample_size, sampler_state, remaining, result):
    """
    Sample a desired number of integers from a range (starting from zero)
    excluding black-listed elements defined in sample state. Used in
    conjunction with `prime_sample_state` method, which initializes state.
    Inspired by Fischer-Yates shuffle.
    """
    # gradually sample from the decreased size range
    for k in range(sample_size):
        i = randrange(remaining)
        result[k] = sampler_state.get(i, i)
        remaining -= 1
        sampler_state[i] = sampler_state.get(remaining, remaining)
        sampler_state.pop(remaining, -1)


@njit(parallel=True)
def mf_random_item_scoring(
    user_factors, item_factors, indptr, indices, size, seedseq, res
):
    """
    Calculate matrix factorization scores over a sample of random items
    excluding the already observed ones.
    """
    num_items, rank = item_factors.shape
    for i in prange(len(indptr) - 1):
        head = indptr[i]
        tail = indptr[i + 1]
        observed = indices[head:tail]
        user_coef = user_factors[i, :]
        set_seed(seedseq[i])  # randomization control for sampling in a thread
        for j, rnd_item in enumerate(sample_unseen(num_items, size, observed)):
            item_coef = item_factors[rnd_item, :]
            tmp = 0
            for k in range(rank):
                tmp += user_coef[k] * item_coef[k]
            res[i, j] = tmp


@njit(parallel=True)
def sample_row_wise(indptr, indices, n_cols, n_samples, seed_seq):
    """
    For every row of a CSR matrix, samples indices not present in this row.
    """
    n_rows = len(indptr) - 1
    result = np.empty((n_rows, n_samples), dtype=indices.dtype)
    for i in prange(n_rows):
        head = indptr[i]
        tail = indptr[i + 1]
        seen_inds = indices[head:tail]
        state = prime_sampler_state(n_cols, seen_inds)
        remaining = n_cols - len(seen_inds)
        set_seed(seed_seq[i])
        sample_fill(n_samples, state, remaining, result[i, :])
    return result


@njit(parallel=True)
def sample_element_wise(indptr, indices, n_cols, n_samples, seed_seq):
    """
    For every nnz entry of a CSR matrix, samples indices not present
    in its corresponding row.
    """
    result = np.empty((indptr[-1], n_samples), dtype=indices.dtype)
    for i in prange(len(indptr) - 1):
        head = indptr[i]
        tail = indptr[i + 1]

        seen_inds = indices[head:tail]
        state = prime_sampler_state(n_cols, seen_inds)
        remaining = n_cols - len(seen_inds)
        set_seed(seed_seq[i])
        for j in range(head, tail):
            sampler_state = state.copy()
            sample_fill(n_samples, sampler_state, remaining, result[j, :])
    return result


@njit
def split_top_continuous(tasks, priorities):
    """
    Sample a sequence of unique tasks of the highest priority ensuring that
    no task will have another instance with the priority level above the lowest priority in the sequence.
    Usecases: avoiding issues with "recommendations from future" when splitting test data by timestamp.
    """
    priority_queue = [(-max(priorities), len(priorities))] # initialize typed
    priority_queue.pop()

    for idx, priority in enumerate(priorities):
        heapq.heappush(priority_queue, (-priority, idx))

    topseq = {}  # continuous sequence of top-priority tasks
    nonseq_idx = []  # top-priority tasks that interrupt continuous sequence

    unique_tasks = set(tasks)
    while unique_tasks:
        _, idx = heapq.heappop(priority_queue)
        task = tasks[idx]
        try:
            visited = topseq[task]
        except:
            unique_tasks.remove(task)
        else:
            nonseq_idx.append(visited)
        topseq[task] = idx

    topseq_idx = [idx for _, idx in topseq.items()]
    lowseq_idx = [idx for _, idx in priority_queue]  # all remaining tasks
    return topseq_idx, lowseq_idx, nonseq_idx