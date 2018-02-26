from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from collections import namedtuple


def no_copy_csr_matrix(data, indices, indptr, shape, dtype):
    # set data and indices manually to avoid index dtype checks
    # and thus prevent possible unnecesssary copies of indices
    matrix = csr_matrix(shape, dtype=dtype)
    matrix.data = data
    matrix.indices = indices
    matrix.indptr = indptr
    return matrix


def build_rank_matrix(recommendations, shape):
    # handle singletone case for a single user
    recommendations = np.array(recommendations, copy=False, ndmin=2)
    n_keys, topn = recommendations.shape
    rank_arr = np.arange(1, topn+1, dtype=np.min_scalar_type(topn))
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
        indptr = np.arange(0, n_keys*topn+1, topn)

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
    # and thus prevent possible unnecesssary copies of indices
    indices = observations[target].values
    indptr = np.r_[0, np.where(np.diff(observations[key].values))[0]+1, n_observations]
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
    # handle singletone case for a single user
    recommendations = np.array(recommendations, copy=False, ndmin=2)
    shape = (recommendations.shape[0], max(recommendations.max(), eval_data[target].max())+1)
    eval_matrix = matrix_from_observations(eval_data, key, target, shape, feedback=feedback)
    eval_matrix_hits, eval_matrix_miss = split_positive(eval_matrix, is_positive)
    rank_matrix = build_rank_matrix(recommendations, shape)
    hits_rank, miss_rank = generate_hits_data(rank_matrix, eval_matrix_hits, eval_matrix_miss)
    return (rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss)


def get_mrr_score(hits_rank):
    return hits_rank.power(-1, 'f8').max(axis=1).mean()


def get_ndcr_discounts(rank_matrix, eval_matrix, topn):
    discounts = np.reciprocal(np.log2(1+rank_matrix.data, dtype='f8'))
    discounts_matrix = rank_matrix._with_data(discounts, copy=False)
    # circumventing problem in ideal_discounts = eval_matrix.tolil()
    # related to incompatible indices dtype
    relevance_per_key = np.array_split(eval_matrix.data, eval_matrix.indptr[1:-1])
    target_id_per_key = np.array_split(eval_matrix.indices, eval_matrix.indptr[1:-1])

    #ideal_indices = [np.argsort(rel)[:-(topn+1):-1] for rel in relevance_per_key]
    #idx = np.arange(2, topn+2)
    ideal_indices = [np.argsort(rel)[::-1] for rel in relevance_per_key]
    idx = np.arange(2, eval_matrix.getnnz(axis=1).max()+2)
    data = np.concatenate([np.reciprocal(np.log2(idx[:len(i)], dtype='f8')) for i in ideal_indices])
    inds = np.concatenate([np.take(r, i) for r, i in zip(target_id_per_key, ideal_indices)])
    ptrs = np.r_[0, np.cumsum([len(i) for i in ideal_indices])]
    ideal_discounts = no_copy_csr_matrix(data, inds, ptrs, eval_matrix.shape, data.dtype)
    return discounts_matrix, ideal_discounts


def get_ndcr_score(eval_matrix, discounts_matrix, ideal_discounts, alternative=False):
    '''Normalized Discounted Cumulative Ranking'''
    if alternative:
        relevance = eval_matrix._with_data(np.exp2(eval_matrix.data)-1, copy=False)
    else:
        relevance = eval_matrix

    dcr = relevance.multiply(discounts_matrix).sum(axis=1)
    idcr = relevance.multiply(ideal_discounts).sum(axis=1)

    with np.errstate(invalid='ignore'):
        score = np.nansum(dcr/idcr) / relevance.shape[0]
    return score


def get_ndcg_score(eval_matrix, discounts_matrix, ideal_discounts, alternative=False):
    '''Normalized Discounted Cumulative Gain'''
    return get_ndcr_score(eval_matrix, discounts_matrix, ideal_discounts, alternative=alternative)


def get_ndcl_score(eval_matrix, discounts_matrix, ideal_discounts, switch_positive, alternative=False):
    '''Normalized Discounted Cumulative Loss'''
    eval_matrix = eval_matrix._with_data(eval_matrix.data-switch_positive, copy=False)
    return get_ndcr_score(eval_matrix, -discounts_matrix, -ideal_discounts, alternative=alternative)


def _get_ranking_scores(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, switch_positive=None, topk=None, alternative=False):
    discounts_matrix, ideal_discounts = get_ndcr_discounts(rank_matrix, eval_matrix, topk)
    ndcg = get_ndcg_score(eval_matrix_hits, discounts_matrix, ideal_discounts, alternative=alternative)
    ndcl = None
    if miss_rank is not None:
        ndcl = get_ndcl_score(eval_matrix_miss, discounts_matrix, ideal_discounts, switch_positive, alternative=alternative)

    ranking_score = namedtuple('Ranking', ['nDCG', 'nDCL'])._make([ndcg, ndcl])
    return ranking_score


def _get_relevance_data(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, not_rated_penalty=None, per_key=False):
    axis = 1 if per_key else None
    true_positive = hits_rank.getnnz(axis=axis)
    if miss_rank is None:
        if not_rated_penalty > 0:
            false_positive = not_rated_penalty * (rank_matrix.getnnz(axis=axis)-true_positive)
        else:
            false_positive = 0
        false_negative = eval_matrix.getnnz(axis=axis) - true_positive
        true_negative = None
    else:
        false_positive = miss_rank.getnnz(axis=axis)
        true_negative = eval_matrix_miss.getnnz(axis=axis) - false_positive
        false_negative = eval_matrix_hits.getnnz(axis=axis) - true_positive
        if not_rated_penalty > 0:
            not_rated = rank_matrix.getnnz(axis=axis)-true_positive-false_positive
            false_positive = false_positive + not_rated_penalty * not_rated
    return [true_positive, false_positive, true_negative, false_negative]


def _get_hits(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, not_rated_penalty=None):
    hits = namedtuple('Hits', ['true_positive', 'false_positive',
                               'true_negative', 'false_negative'])
    hits = hits._make(_get_relevance_data(rank_matrix, hits_rank, miss_rank,
                                          eval_matrix, eval_matrix_hits, eval_matrix_miss,
                                          not_rated_penalty, False))
    return hits


def _get_relevance_scores(rank_matrix, hits_rank, miss_rank, eval_matrix, eval_matrix_hits, eval_matrix_miss, not_rated_penalty=None):
    [true_positive, false_positive,
     true_negative, false_negative] = _get_relevance_data(rank_matrix, hits_rank, miss_rank,
                                                          eval_matrix, eval_matrix_hits, eval_matrix_miss,
                                                          not_rated_penalty, True)

    with np.errstate(invalid='ignore'):
        # true positive rate
        precision = true_positive / (true_positive + false_positive)
        # sensitivity
        recall = true_positive / (true_positive + false_negative)
        # false negative rate
        miss_rate = false_negative / (false_negative + true_positive)
        if true_negative is not None:
            # false positive rate
            fallout = false_positive / (false_positive + true_negative)
            # true negative rate
            specifity = true_negative / (false_positive + true_negative)
        else:
            fallout = specifity = None

    n_keys = hits_rank.shape[0]
    #average over all users
    precision = np.nansum(precision) / n_keys
    recall = np.nansum(recall) / n_keys
    miss_rate = np.nansum(miss_rate) / n_keys
    if true_negative is not None:
        specifity = np.nansum(specifity) / n_keys
        fallout = np.nansum(fallout) / n_keys

    scores = namedtuple('Relevance', ['precision', 'recall', 'fallout', 'specifity', 'miss_rate'])
    scores = scores._make([precision, recall, fallout, specifity, miss_rate])
    return scores


def unmask(x):
    # return `None` instead of single  `mask` value
    return None if x is np.ma.masked else x


def get_hits(matched_predictions, positive_feedback, not_rated_penalty):
    reldata = get_relevance_data(matched_predictions, positive_feedback, not_rated_penalty)
    true_pos, false_pos = reldata.tp, reldata.fp
    true_neg, false_neg = reldata.tn, reldata.fn

    true_pos_hits = unmask(true_pos.sum())
    false_pos_hits = unmask(false_pos.sum())
    true_neg_hits = unmask(true_neg.sum())
    false_neg_hits = unmask(false_neg.sum())

    hits = namedtuple('Hits', ['true_positive', 'true_negative', 'false_positive', 'false_negative'])
    hits = hits._make([true_pos_hits, true_neg_hits, false_pos_hits, false_neg_hits])
    return hits


def get_relevance_scores(matched_predictions, positive_feedback, not_rated_penalty):
    users_num = matched_predictions.shape[0]
    reldata = get_relevance_data(matched_predictions, positive_feedback, not_rated_penalty)
    true_pos, false_pos = reldata.tp, reldata.fp
    true_neg, false_neg = reldata.tn, reldata.fn

    with np.errstate(invalid='ignore'):
        # true positive rate
        precision = true_pos / (true_pos + false_pos)
        # sensitivity
        recall = true_pos / (true_pos + false_neg)
        # false positive rate
        fallout = false_pos / (false_pos + true_neg)
        # true negative rate
        specifity = true_neg / (false_pos + true_neg)
        # false negative rate
        miss_rate = false_neg / (false_neg + true_pos)

    #average over all users
    precision = unmask(np.nansum(precision) / users_num)
    recall = unmask(np.nansum(recall) / users_num)
    fallout = unmask(np.nansum(fallout) / users_num)
    specifity = unmask(np.nansum(specifity) / users_num)
    miss_rate = unmask(np.nansum(miss_rate) / users_num)

    scores = namedtuple('Relevance', ['precision', 'recall', 'fallout', 'specifity', 'miss_rate'])
    scores = scores._make([precision, recall, fallout, specifity, miss_rate])
    return scores


def get_ranking_scores(matched_predictions, feedback_data, switch_positive, alternative=True):
    users_num, topk, holdout = matched_predictions.shape
    ideal_scores_idx = np.argsort(feedback_data, axis=1)[:, ::-1] #returns column index only
    ideal_scores_idx = np.ravel_multi_index((np.arange(feedback_data.shape[0])[:, None], ideal_scores_idx), dims=feedback_data.shape)

    where = np.ma.where if np.ma.is_masked(feedback_data) else np.where
    is_positive = feedback_data >= switch_positive
    positive_feedback = where(is_positive, feedback_data, 0)
    negative_feedback = where(~is_positive, feedback_data-switch_positive, 0)

    relevance_scores_pos = (matched_predictions * positive_feedback[:, None, :]).sum(axis=2)
    relevance_scores_neg = (matched_predictions * negative_feedback[:, None, :]).sum(axis=2)
    ideal_scores_pos = positive_feedback.ravel()[ideal_scores_idx]
    ideal_scores_neg = negative_feedback.ravel()[ideal_scores_idx]

    if alternative:
        relevance_scores_pos = 2**relevance_scores_pos - 1
        relevance_scores_neg = 2.0**relevance_scores_neg - 1
        ideal_scores_pos = 2**ideal_scores_pos - 1
        ideal_scores_neg = 2.0**ideal_scores_neg - 1

    disc_num = max(topk, holdout)
    discount = np.log2(np.arange(2, disc_num+2))
    dcg = (relevance_scores_pos / discount[:topk]).sum(axis=1)
    dcl = (relevance_scores_neg / -discount[:topk]).sum(axis=1)
    idcg = (ideal_scores_pos / discount[:holdout]).sum(axis=1)
    idcl = (ideal_scores_neg / -discount[:holdout]).sum(axis=1)

    with np.errstate(invalid='ignore'):
        ndcg = unmask(np.nansum(dcg / idcg) / users_num)
        ndcl = unmask(np.nansum(dcl / idcl) / users_num)

    ranking_score = namedtuple('Ranking', ['nDCG', 'nDCL'])._make([ndcg, ndcl])
    return ranking_score


def get_relevance_data(matched_items, positive_feedback, not_rated_penalty):
    negative_feedback = ~positive_feedback
    missed_items = ~matched_items

    #relevant items present both in preferences and in recommendations
    in_top_pos = (matched_items & positive_feedback[:, None, :]).any(axis=2)
    #irrelevant items present both in preferences and in recommendations
    in_top_neg = (matched_items & negative_feedback[:, None, :]).any(axis=2)
    #irrelevant items, present in preferences and not recommended by algo
    no_recom_pos = (missed_items & negative_feedback[:, None, :]).all(axis=1)
    #relevant items, present in preferences and not recommended by algo
    no_recom_neg = (missed_items & positive_feedback[:, None, :]).all(axis=1)

    true_pos = in_top_pos.sum(axis=1)
    false_pos = in_top_neg.sum(axis=1)
    true_neg = no_recom_pos.sum(axis=1)
    false_neg = no_recom_neg.sum(axis=1)

    if not_rated_penalty > 0:
        #penalize prediction of items not present in preferences (i.e. not rated yet)
        #this decreases the advantage of the tensor algorithm,
        #as it penalizes potentially relevant predictions, taking into account
        #that we are able to correctly predict highly rated items in ~95% cases
        #this option should not be turned on
        not_rated_items = missed_items.all(axis=2)
        false_pos = false_pos + not_rated_penalty * not_rated_items.sum(axis=1)

    relevance_data = namedtuple('RelevanceData', ['tp', 'fp', 'tn', 'fn'])
    relevance_data = relevance_data._make([true_pos, false_pos, true_neg, false_neg])
    return relevance_data
