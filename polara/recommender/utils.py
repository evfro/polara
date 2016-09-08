from __future__ import division
import numpy as np
from polara.tools.systools import get_available_memory


MEMORY_HARD_LIMIT = 1 # in gigbytes, default=1
# varying this value may significantly impact performance
# setting it to None or large value typically reduces performance,
# as iterating over a smaller number of huge arrays takes longer
# than over a higher number of smaller arrays


def range_division(length, fit_size):
    # based on np.array_split
    n_chunks = length // fit_size + int((length % fit_size)>0)
    chunk_size, remainder =  divmod(length, n_chunks)
    chunk_sizes = ([0] + remainder * [chunk_size+1] +
                   (n_chunks-remainder) * [chunk_size])
    return np.cumsum(chunk_sizes)


def get_chunk_size(shp, topk, td_multiplier):
    chunk_size = shp[0]
    #dealing with huge sizes (typically in tensor case)
    shp = [s/1024 if i < 2 else s for i, s in enumerate(shp)]
    itemsize_scores = np.dtype(np.float64).itemsize / 1024
    itemsize_topk = np.dtype(np.int64).itemsize / 1024

    scores_memory = np.prod(shp[:2]) * td_multiplier * itemsize_scores # in gigbytes
    topk_memory = shp[0] * (topk/1024) * itemsize_topk # in gigbytes

    # take no more than 80% of available memory
    memory_limit = 0.8 * get_available_memory()
    if MEMORY_HARD_LIMIT:
        # too large arrays create significant overhead (with dot or tensordot)
        memory_limit = min(memory_limit, MEMORY_HARD_LIMIT)
    required_memory = scores_memory + topk_memory # memory at peak usage
    if required_memory > memory_limit:
        chunk_size = min(int((memory_limit - topk_memory) /
                             (shp[1]*itemsize_scores*(td_multiplier/1024) +
                              itemsize_topk/(1024**2)) - 1),
                         chunk_size)
        if chunk_size <= 0:
            # potentially raises error even if there's enough memory
            # due to hard limit (for really large files)
            raise MemoryError()
    return chunk_size


def array_split(shp, topk, multiplier):
    chunk_size = get_chunk_size(shp, topk, multiplier)
    split = range_division(shp[0], chunk_size)
    return split
