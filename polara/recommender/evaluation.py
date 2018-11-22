from collections import namedtuple

import numpy as np
from scipy.sparse import csr_matrix


def no_copy_csr_matrix(data, indices, indptr, shape, dtype):
    # set data and indices manually to avoid index dtype checks
    # and thus prevent possible unnecessary copies of indices
    matrix = csr_matrix(shape, dtype=dtype)
    matrix.data = data
    matrix.indices = indices
    matrix.indptr = indptr
    return matrix


def safe_divide(a, b, mask=None, dtype=None):
    pos = mask if mask is not None else a > 0
    res = np.zeros(len(a), dtype=dtype)
    np.divide(a, b, where=pos, out=res)
    return res


def build_rank_matrix(recommendations, shape):
    # handle singleton case for a single user
    recommendations = np.array(recommendations, copy=False, ndmin=2)
    n_keys, topn = recommendations.shape
    rank_arr = np.arange(1, topn + 1, dtype=np.min_scalar_type(topn))
    recs_rnk = np.lib.stride_tricks.as_strided(rank_arr, (n_keys, topn), (0, rank_arr.itemsize))
    # support models that may generate < top-n recommendations
    # such models generate self._pad_const, which is negative by convention
    valid_recommendations = recommendations >= 0
    if not valid_recommendations.all():
        data = recs_rnk[valid_recommendations]
        indices = recommendations[valid_recommendations]
        indptr = np.r_[0, np.cumsum(valid_recommendations.sum(axis=1))]
    else:
        data = recs_rnk.ravel()
        indices = recommendations.ravel()
        indptr = np.arange(0, n_keys * topn + 1, topn)

    rank_matrix = no_copy_csr_matrix(data, indices, indptr, shape, rank_arr.dtype)
    return rank_matrix


def matrix_from_observations(observations, key, target, shape, feedback=None):
    # assumes that observations dataframe and recommendations matrix
    # are aligned on and sorted by the "key"
    n_observations = observations.shape[0]
    if feedback:
        data = observations[feedback].values
        dtype = data.dtype
    else:
        dtype = np.bool
        data = np.ones(n_observations, dtype=dtype)
    # set data and indices manually to avoid index dtype checks
    # and thus prevent possible unnecessary copies of indices
    indices = observations[target].values
    indptr = np.r_[0, np.where(np.diff(observations[key].values))[0] + 1, n_observations]
    matrix = no_copy_csr_matrix(data, indices, indptr, shape, dtype)
    return matrix


def split_positive(eval_matrix, is_positive):
    if is_positive is not None:
        eval_matrix_hits = eval_matrix.copy()
        eval_matrix_hits.data[~is_positive] = 0
        eval_matrix_hits.eliminate_zeros()

        eval_matrix_miss = eval_matrix.copy()
        eval_matrix_miss.data[is_positive] = 0
        eval_matrix_miss.eliminate_zeros()
    else:
        eval_matrix_hits = eval_matrix
        eval_matrix_miss = None
    return eval_matrix_hits, eval_matrix_miss


def generate_hits_data(rank_matrix, eval_matrix_hits, eval_matrix_miss=None):
    # Note: scipy logical operations (OR, XOR, AND) are not supported yet
    # see https://github.com/scipy/scipy/pull/5411
    dtype = np.bool
    hits_rank = eval_matrix_hits._with_data(eval_matrix_hits.data.astype(dtype, copy=False), copy=False).multiply(rank_matrix)
    miss_rank = None
    if eval_matrix_miss is not None:
        miss_rank = eval_matrix_miss._with_data(eval_matrix_miss.data.astype(dtype, copy=False), copy=False).multiply(rank_matrix)
    return hits_rank, miss_rank


def assemble_scoring_matrices(recommendations, eval_data, key, target, is_positive, feedback=None):
    # handle singleton case for a single user
    recommendations = np.array(recommendations, copy=False, ndmin=2)
    shape = (recommendations.shape[0], max(recommendations.max(), eval_data[target].max()) + 1)
    eval_matrix = matrix_from_observations(eval_data, key, target, shape, feedback=feedback)
    eval_matrix_hits, eval_matrix_miss = split_positive(eval_matrix, is_positive)
    rank_matrix = build_rank_matrix(recommendations, shape)
    hits_rank, miss_rank = generate_hits_data(rank_matrix, eval_matrix_hits, eval_matrix_miss)
    return (rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss)


def get_hr_score(hits_rank):
    'Hit-Rate score'
    hr = hits_rank.getnnz(axis=1).mean()
    return namedtuple('Relevance', ['hr'])._make([hr])


def get_mrr_score(hits_rank):
    'Mean Reciprocal Rank score'
    mrr = hits_rank.power(-1, 'f8').max(axis=1).mean()
    return namedtuple('Ranking', ['mrr'])._make([mrr])


def get_ndcr_discounts(rank_matrix, eval_matrix, topn):
    discounts = np.reciprocal(np.log2(1 + rank_matrix.data, dtype='f8'))
    discounts_matrix = rank_matrix._with_data(discounts, copy=False)
    # circumventing problem in ideal_discounts = eval_matrix.tolil()
    # related to incompatible indices dtype
    relevance_per_key = np.array_split(eval_matrix.data, eval_matrix.indptr[1:-1])
    target_id_per_key = np.array_split(eval_matrix.indices, eval_matrix.indptr[1:-1])

    # ideal_indices = [np.argsort(rel)[:-(topn+1):-1] for rel in relevance_per_key]
    # idx = np.arange(2, topn+2)
    ideal_indices = [np.argsort(rel)[::-1] for rel in relevance_per_key]
    idx = np.arange(2, eval_matrix.getnnz(axis=1).max() + 2)
    data = np.concatenate([np.reciprocal(np.log2(idx[:len(i)], dtype='f8')) for i in ideal_indices])
    inds = np.concatenate([np.take(r, i) for r, i in zip(target_id_per_key, ideal_indices)])
    ptrs = np.r_[0, np.cumsum([len(i) for i in ideal_indices])]
    ideal_discounts = no_copy_csr_matrix(data, inds, ptrs, eval_matrix.shape, data.dtype)
    return discounts_matrix, ideal_discounts


def get_ndcr_score(eval_matrix, discounts_matrix, ideal_discounts, alternative=False):
    '''Normalized Discounted Cumulative Ranking'''
    if alternative:
        relevance = eval_matrix._with_data(np.exp2(eval_matrix.data) - 1, copy=False)
    else:
        relevance = eval_matrix
    dcr = np.array(relevance.multiply(discounts_matrix).sum(axis=1), copy=False).squeeze()
    idcr = np.array(relevance.multiply(ideal_discounts).sum(axis=1), copy=False).squeeze()
    return safe_divide(dcr, idcr).mean()


def get_ndcg_score(eval_matrix, discounts_matrix, ideal_discounts, alternative=False):
    '''Normalized Discounted Cumulative Gain'''
    return get_ndcr_score(eval_matrix, discounts_matrix, ideal_discounts, alternative=alternative)


def get_ndcl_score(eval_matrix, discounts_matrix, ideal_discounts, switch_positive, alternative=False):
    '''Normalized Discounted Cumulative Loss'''
    eval_matrix = eval_matrix._with_data(eval_matrix.data - switch_positive, copy=False)
    return get_ndcr_score(eval_matrix, -discounts_matrix, -ideal_discounts, alternative=alternative)


def get_ranking_scores(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, switch_positive=None, topk=None, alternative=False):
    discounts_matrix, ideal_discounts = get_ndcr_discounts(rank_matrix, eval_matrix, topk)
    ndcg = get_ndcg_score(eval_matrix_hits, discounts_matrix, ideal_discounts, alternative=alternative)
    ndcl = None
    if miss_rank is not None:
        ndcl = get_ndcl_score(eval_matrix_miss, discounts_matrix, ideal_discounts, switch_positive, alternative=alternative)

    ranking_score = namedtuple('Ranking', ['nDCG', 'nDCL'])._make([ndcg, ndcl])
    return ranking_score


def get_relevance_data(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, not_rated_penalty=None, per_key=False):
    axis = 1 if per_key else None
    true_positive = hits_rank.getnnz(axis=axis)
    if miss_rank is None:
        if not_rated_penalty > 0:
            false_positive = not_rated_penalty * (rank_matrix.getnnz(axis=axis) - true_positive)
        else:
            false_positive = 0
        false_negative = eval_matrix.getnnz(axis=axis) - true_positive
        true_negative = None
    else:
        false_positive = miss_rank.getnnz(axis=axis)
        true_negative = eval_matrix_miss.getnnz(axis=axis) - false_positive
        false_negative = eval_matrix_hits.getnnz(axis=axis) - true_positive
        if not_rated_penalty > 0:
            not_rated = rank_matrix.getnnz(axis=axis) - true_positive - false_positive
            false_positive = false_positive + not_rated_penalty * not_rated
    return [true_positive, false_positive, true_negative, false_negative]


def get_hits(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, not_rated_penalty=None):
    hits = namedtuple('Hits', ['true_positive', 'false_positive',
                               'true_negative', 'false_negative'])
    hits = hits._make(get_relevance_data(rank_matrix, hits_rank, miss_rank,
                                         eval_matrix, eval_matrix_hits, eval_matrix_miss,
                                         not_rated_penalty, False))
    return hits


def get_relevance_scores(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, not_rated_penalty=None):
    [true_positive, false_positive,
     true_negative, false_negative] = get_relevance_data(rank_matrix, hits_rank, miss_rank,
                                                         eval_matrix, eval_matrix_hits, eval_matrix_miss,
                                                         not_rated_penalty, True)
    # non-zero mask for safe division
    tpnz = true_positive > 0
    fnnz = false_negative > 0
    # true positive rate
    precision = safe_divide(true_positive, true_positive + false_positive, tpnz).mean()
    # sensitivity
    recall = safe_divide(true_positive, true_positive + false_negative, tpnz).mean()
    # false negative rate
    miss_rate = safe_divide(false_negative, false_negative + true_positive, fnnz).mean()

    if true_negative is not None:
        # non-zero mask for safe division
        fpnz = false_positive > 0
        tnnz = true_negative > 0
        # false positive rate
        fallout = safe_divide(false_positive, false_positive + true_negative, fpnz).mean()
        # true negative rate
        specifity = safe_divide(true_negative, false_positive + true_negative, tnnz).mean()
    else:
        fallout = specifity = None

    scores = namedtuple('Relevance', ['precision', 'recall', 'fallout', 'specifity', 'miss_rate'])
    scores = scores._make([precision, recall, fallout, specifity, miss_rate])
    return scores


def get_experience_scores(recommendations, total):
    cov = len(np.unique(recommendations)) / total
    scores = namedtuple('Experience', ['coverage'])._make([cov])
    return scores
