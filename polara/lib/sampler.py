from random import randrange
from random import seed as set_seed
from numba import njit, prange

# Sampler below is an adapted "jit-friendly" version of the accepted answer at
# https://stackoverflow.com/questions/18921302/how-to-incrementally-sample-without-replacement
@njit
def sample_unseen(n, size, exclude):
    '''
    This is a generator to sample a desired number of integers from a range
    (starting from zero) excluding black-listed elements. It samples items
    one by one, which makes it memory efficient and convenient for "on-the-fly"
    calculations with sampled elements.
    '''
    # initialize dicts with desired type
    state = {n: n} # n will never get sampled, can safely use
    track = {n: n}
    # reindex excluded items, placing them in the end
    for i, item in enumerate(exclude):
        pos = n - i - 1
        x = track.get(item, item)
        t = state.get(pos, pos)
        state[x] = t
        track[t] = x
        state.pop(pos, n)
        track.pop(item, n)
    track = None
    # ensure excluded items are not sampled
    remaining = n - len(exclude)
    # gradually sample from the decreased size range
    for _ in range(size):
        i = randrange(remaining)
        yield state[i] if i in state else i # avoid numba bug with dict.get(i,i)
        remaining -= 1
        state[i] = state.get(remaining, remaining)
        state.pop(remaining, n)


@njit(parallel=True)
def mf_random_item_scoring(user_factors, item_factors, indptr, indices,
                           size, seedseq, res):
    '''
    Calculate matrix factorization scores over a sample of random items
    excluding the already observed ones.
    '''
    num_items, rank = item_factors.shape
    for i in prange(len(indptr)-1):
        head = indptr[i]
        tail = indptr[i+1]
        observed = indices[head:tail]
        user_coef = user_factors[i, :]
        set_seed(seedseq[i]) # randomization control for sampling in a thread
        for j, rnd_item in enumerate(sample_unseen(num_items, size, observed)):
            item_coef = item_factors[rnd_item, :]
            tmp = 0
            for k in range(rank):
                tmp += user_coef[k] * item_coef[k]
            res[i, j] = tmp
