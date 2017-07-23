from __future__ import division
import sys
import numpy as np
from polara.tools.systools import get_available_memory


MEMORY_HARD_LIMIT = 1 # in gigbytes, default=1, depends on hardware
# varying this value may significantly impact performance
# setting it to None or large value typically reduces performance,
# as iterating over a smaller number of huge arrays takes longer
# than over a higher number of smaller arrays

tuplsize = sys.getsizeof(())
itemsize = np.dtype(np.intp).itemsize
pntrsize = sys.getsizeof(1.0)
# size of list of tuples of indices - to estimate when to convert sparse matrix to dense
# based on http://stackoverflow.com/questions/15641344/python-memory-consumption-dict-vs-list-of-tuples
# and https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
NNZ_MAX = int(MEMORY_HARD_LIMIT * (1024**3) / (tuplsize + 2*(pntrsize + itemsize)))


def range_division(length, fit_size):
    # based on np.array_split
    n_chunks = length // fit_size + int((length % fit_size)>0)
    chunk_size, remainder =  divmod(length, n_chunks)
    chunk_sizes = ([0] + remainder * [chunk_size+1] +
                   (n_chunks-remainder) * [chunk_size])
    return np.cumsum(chunk_sizes)


def get_chunk_size(shp, result_width, scores_multiplier, dtypes=None):
    chunk_size = shp[0]
    #dealing with huge sizes (typically in tensor case)
    shp = [s/1024 if i < 2 else s for i, s in enumerate(shp)]

    if dtypes:
        result_itemsize = np.dtype(dtypes[0]).itemsize / 1024
        scores_itemsize = np.dtype(dtypes[1]).itemsize / 1024
    else:
        # standard case: result is matrix of indices, intermediate scores are floats
        result_itemsize = np.dtype(np.int64).itemsize / 1024
        scores_itemsize = np.dtype(np.float64).itemsize / 1024

    result_memory = shp[0] * (result_width/1024) * result_itemsize # in gigbytes
    scores_memory = np.prod(shp[:2]) * scores_multiplier * scores_itemsize # in gigbytes

    # take no more than 80% of available memory
    memory_limit = 0.8 * get_available_memory()
    if MEMORY_HARD_LIMIT:
        # too large arrays create significant overhead (with dot or tensordot)
        memory_limit = min(memory_limit, MEMORY_HARD_LIMIT)
    required_memory = scores_memory + result_memory # memory at peak usage
    if required_memory > memory_limit:
        chunk_size = min(int((memory_limit - result_memory) /
                             (shp[1]*scores_itemsize*(scores_multiplier/1024) +
                              result_itemsize/(1024**2)) - 1),
                         chunk_size)
        if chunk_size <= 0:
            # potentially raises error even if there's enough memory
            # due to hard limit (for really large files)
            raise MemoryError()
    return chunk_size


def array_split(shp, result_width, scores_multiplier, dtypes=None):
    chunk_size = get_chunk_size(shp, result_width, scores_multiplier, dtypes=dtypes)
    split = range_division(shp[0], chunk_size)
    return split
