from io import BytesIO
from urllib.request import urlopen
import numpy as np
from polara.tools.systools import get_available_memory
from polara.recommender import defaults

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
    if defaults.memory_hard_limit:
        # too large arrays create significant overhead (with dot or tensordot)
        memory_limit = min(memory_limit, defaults.memory_hard_limit)
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


def read_npz_form_url(url, allow_pickle=False):
    '''Read numpy's .npz file directly from source url.'''
    with urlopen(url) as response:
        file_handle = BytesIO(response.read())
        return np.load(file_handle, allow_pickle=allow_pickle)
