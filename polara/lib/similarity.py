# python 2/3 interoperability
from __future__ import division

try:
    range = xrange
except NameError:
    pass

try:
    long
except NameError:
    long = int

import math
import types
from collections import defaultdict, OrderedDict
import numpy as np
from numba import jit
import scipy as sp
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, SparseEfficiencyWarning
import warnings


def _fix_empty_features(feature_mat):
    if feature_mat.format != 'csc':
        raise NotImplementedError
    nf = feature_mat.getnnz(axis=1)
    zf = (nf == 0)
    if zf.any():
        # add +1 dummy feature for every zero row
        # to avoid zeros on diagonal of similarity matrix
        nnz = zf.sum(dtype=long) #long is required to avoid overflow with int32 dtype
        nnz_ind = np.where(zf)[0]
        feature_mat.indptr = np.hstack([feature_mat.indptr[:-1], feature_mat.indptr[-1]+np.arange(nnz+1)])
        feature_mat.indices = np.hstack([feature_mat.indices, nnz_ind])
        feature_mat.data = np.hstack([feature_mat.data, np.ones((nnz,), dtype=feature_mat.data.dtype)])
        feature_mat._shape = (feature_mat.shape[0], feature_mat.shape[1]+nnz)
        nf[nnz_ind] = 1
    return nf


def set_diagonal_values(mat, val=1):
    # disable warning when setting diagonal elements of sparse similarity matrix
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=SparseEfficiencyWarning)
        mat.setdiag(val)


def safe_inverse_root(d, dtype=None):
    pos_d = d > 0
    res = np.zeros(len(d), dtype=dtype)
    np.power(d, -0.5, where=pos_d, dtype=dtype, out=res)
    return res


def normalize_binary_features(feature_mat, dtype=None):
    sqsum = feature_mat.getnnz(axis=1)
    if feature_mat.format == 'csc':
        ind = feature_mat.indices.copy()
        ptr = feature_mat.indptr.copy()
        norm_data = 1 / np.sqrt(np.take(sqsum, ind))
        normed = csc_matrix((norm_data, ind, ptr), shape=feature_mat.shape, dtype=dtype)
    else:
        norm_data = safe_inverse_root(sqsum, dtype=dtype)
        normed = sp.sparse.diags(norm_data).dot(feature_mat)
    return normed


def normalize_features(feature_mat, dtype=None):
    if feature_mat.format == 'csc':
        ptr = feature_mat.indptr.copy()
        ind = feature_mat.indices.copy()
        sqsum = np.bincount(ind, weights=feature_mat.data**2)
        # could do np.add.at(sqsum, indices, data**2), but bincount seems faster
        norm_data = feature_mat.data / np.sqrt(np.take(sqsum, ind))
        normed = csc_matrix((norm_data, ind, ptr), shape=feature_mat.shape, dtype=dtype)
    else:
        sqsum = feature_mat.power(2).sum(axis=1).view(type=np.ndarray).squeeze()
        # avoid zero division
        norm_data = safe_inverse_root(sqsum, dtype=dtype)
        normed = sp.sparse.diags(norm_data).dot(feature_mat)
    return normed


def tfidf_transform(feature_mat):
    itemfreq = 1 + feature_mat.getnnz(axis=0)
    N = 1 + feature_mat.shape[0]
    idf = np.log(N/itemfreq)

    if feature_mat.format == 'csr':
        ind = feature_mat.indices
        ptr = feature_mat.indptr
        data = np.take(idf, ind)
        transformed = csr_matrix((data, ind, ptr), shape=feature_mat.shape)
    else:
        # safe multiplicationin in case data is not binary
        transformed = (feature_mat!=0).dot(sp.sparse.diags(idf))
    return transformed


@jit(nopython=True)
def _jaccard_similarity_inplace(sdata, indcs, pntrs, nf):
    # assumes data is symmetric => format doesn't matter
    # it doesn't enforce  1 on the diagonal!
    # so the similarity data must be prepared accordingly
    ncols = len(pntrs) - 1
    for col in range(ncols):
        lind = pntrs[col]
        rind = pntrs[col+1]
        nf_col = nf[col]
        for j in range(lind, rind):
            row = indcs[j]
            denom = nf_col + nf[row] - sdata[j]
            sdata[j] /= denom


def jaccard_similarity(F, fill_diagonal=True):
    F = (F!=0)
    nf = F.getnnz(axis=1)

    S = F.dot(F.T).astype(np.float64) # dtype needed for inplace jaccard index computation
    _jaccard_similarity_inplace(S.data, S.indices, S.indptr, nf)
    if fill_diagonal:
        # eigenvalues computation is very sensitive to roundoff errors in
        # similarity matrix; explicitly seting diagonal values is better
        # than using _fix_empty_features
        set_diagonal_values(S, 1)
    return S


def cosine_similarity(F, fill_diagonal=True, assume_binary=False):
    normalize = normalize_binary_features if assume_binary else normalize_features
    F = normalize(F)
    S = F.dot(F.T)
    if fill_diagonal:
        # eigenvalues computation is very sensitive to roundoff errors in
        # similarity matrix; explicitly seting diagonal values is better
        # than using _fix_empty_features
        set_diagonal_values(S, 1)
    return S


def cosine_tfidf_similarity(F, fill_diagonal=True):
    F = tfidf_transform(F)
    S = cosine_similarity(F, fill_diagonal=fill_diagonal, assume_binary=False)
    return S


@jit(nopython=True)
def _jaccard_similarity_weighted_tri(dat, ind, ptr, shift):
    z = dat[0] #np.float64
    # need to initialize lists with certain dtype
    data = [z]
    cols = [z]
    rows = [z]

    nrows = len(ptr) - 1
    for i in range(nrows):
        lind_i = ptr[i]
        rind_i = ptr[i+1]
        if lind_i != rind_i:
            ish = i + shift
            for j in range(ish, nrows):
                lind_j = ptr[j]
                rind_j = ptr[j+1]
                min_sum = 0
                max_sum = 0
                for k in range(lind_j, rind_j):
                    for s in range(lind_i, rind_i):
                        iind = ind[s]
                        jind = ind[k]
                        if iind == jind:
                            min_val = min(dat[s], dat[k])
                            max_val = max(dat[s], dat[k])
                            min_sum += min_val
                            max_sum += max_val
                            break
                    else:
                        max_sum += dat[k]

                for s in range(lind_i, rind_i):
                    for k in range(lind_j, rind_j):
                        iind = ind[s]
                        jind = ind[k]
                        if iind == jind:
                            break
                    else:
                        max_sum += dat[s]

                if min_sum:
                    wjac = min_sum / max_sum
                    data.append(wjac)
                    cols.append(i)
                    rows.append(j)

    return data[1:], rows[1:], cols[1:] #ignore initialization element


def jaccard_similarity_weighted(F, fill_diagonal=True):
    assert F.format == 'csr'
    if not F.has_sorted_indices:
        F.sort_indices()

    ind = F.indices
    ptr = F.indptr
    dat = F.data.astype(np.float64, copy=False) # dtype needed for jaccard computation

    shift = 1 if fill_diagonal else 0
    data, rows, cols = _jaccard_similarity_weighted_tri(dat, ind, ptr, shift)

    S = coo_matrix((data, (rows, cols)), shape=(F.shape[0],)*2).tocsc()
    S += S.T # doubles diagonal values if fill_diagonal is False

    if fill_diagonal:
        set_diagonal_values(S, 1)
    else:
        set_diagonal_values(S, np.sign(S.diagonal())) # set to 1, preserve zeros
    return S


def jaccard_similarity_weighted_dense(F, fill_diagonal=True):
    if (F.data < 0).any():
        raise ValueError

    f = F.sum(axis=1).view(type=np.ndarray).squeeze()
    fplus = f + f[:, None]
    FA = F.A
    fminus = np.abs(FA[None, :, :] - FA[:, None, :]).sum(axis=2)

    with np.errstate(invalid='ignore'):
        res = np.nan_to_num((fplus - fminus) / (fplus + fminus))

    if fill_diagonal:
        np.fill_diagonal(res, 1)
    return res


def uniquify_ordered(seq):
    # order preserving
    # https://www.peterbe.com/plog/uniqifiers-benchmark
    # http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def build_indicator_matrix(labels, max_items=None):
    indices = [i for x in labels for i in x]
    indprt = np.r_[0, labels.apply(len).cumsum().values]
    data = np.ones_like(indices, dtype=np.bool)
    shape = [len(labels), max_items or indices.max()+1]
    return csr_matrix((data, indices, indprt), shape=shape)


def feature2sparse(feature_data, ranking=None, deduplicate=True, labels=None):
    if deduplicate:
        feature_data = feature_data.apply(uniquify_ordered if ranking else set)

    if labels:
        feature_lbl = labels
        indices = []
        indlens = []
        for items in feature_data:
            # wiil also remove unknown items to ensure index consistency
            inds = [feature_lbl[item] for item in items if item in feature_lbl]
            indices.extend(inds)
            indlens.append(len(inds))
    else:
        feature_lbl = defaultdict(lambda: len(feature_lbl))
        indices = [feature_lbl[item] for items in feature_data for item in items]
        indlens = feature_data.apply(len).values

    indptr = np.r_[0, np.cumsum(indlens)]

    if ranking:
        if ranking is True:
            ranking = 'linear'

        if isinstance(ranking, str):
            if ranking.lower() == 'linear':
                data = [1/(n+1) for items in feature_data for n, item in enumerate(items)]
            elif ranking.lower() == 'exponential':
                data = [math.exp(-n) for items in feature_data for n, item in enumerate(items)]
            elif ranking.lower() == 'bag-of-features':
                raise NotImplementedError
            else:
                raise ValueError
        elif isinstance(ranking, types.FunctionType):
            data = [ranking(n) for items in feature_data for n, item in enumerate(items)]
    else:
        data = np.ones_like(indices)

    feature_mat = csr_matrix((data, indices, indptr),
                             shape=(feature_data.shape[0], len(feature_lbl)))
    if not ranking:
        feature_mat = feature_mat.tocsc()

    return feature_mat, dict(feature_lbl)


def get_features_data(meta_data, ranking=None, deduplicate=True, labels=None):
    feature_mats = OrderedDict()
    feature_lbls = OrderedDict()
    features = meta_data.columns

    ranking = ranking or {}

    if ranking is True:
        ranking = 'linear'

    if isinstance(ranking, str):
        ranking = [ranking] * len(features)

    if not isinstance(ranking, dict):
        ranking = {k: v for k, v in zip(features, ranking)}

    for feature in features:
        feature_data = meta_data[feature]
        mat, lbl = feature2sparse(feature_data,
                                  ranking=ranking.get(feature, None),
                                  deduplicate=deduplicate,
                                  labels=labels[feature] if labels else None)
        feature_mats[feature], feature_lbls[feature] = mat, lbl
    return feature_mats, feature_lbls


def stack_features(features, add_identity=False, normalize=True, dtype=None, labels=None, stacked_index=False, **kwargs):
    feature_mats, feature_lbls = get_features_data(features, labels=labels, **kwargs)

    all_matrices = list(feature_mats.values())
    if add_identity:
        identity = sp.sparse.eye(features.shape[0])
        all_matrices = [identity] + all_matrices

    stacked_features = sp.sparse.hstack(all_matrices, format='csr', dtype=dtype)

    if normalize:
        norm = stacked_features.getnnz(axis=1)
        norm = norm.astype(np.promote_types(norm.dtype, 'f4'))
        scaling = np.power(norm, -1, where=norm>0, dtype=dtype)
        stacked_features = sp.sparse.diags(scaling).dot(stacked_features)

    if stacked_index:
        index_shift = identity.shape[1] if add_identity else 0
        for feature, lbls in feature_lbls.items():
            feature_lbls[feature] = {k:v+index_shift for k, v in lbls.items()}
            index_shift += feature_mats[feature].shape[1]
    return stacked_features, feature_lbls


def _sim_func(func_type):
    if func_type.lower() == 'jaccard':
        return jaccard_similarity
    elif func_type.lower() == 'cosine':
        return cosine_similarity
    elif func_type.lower() == 'tfidf-cosine':
        return cosine_tfidf_similarity
    elif func_type.lower() == 'jaccard-weighted':
        return jaccard_similarity_weighted
    else:
        raise NotImplementedError


def one_hot_similarity(meta_data):
    raise NotImplementedError


def get_similarity_data(meta_data, similarity_type='jaccard'):
    features = meta_data.columns

    if isinstance(similarity_type, str):
        similarity_type = [similarity_type] * len(features)

    if not isinstance(similarity_type, dict):
        similarity_type = {k: v for k, v in zip(features, similarity_type)}

    similarity_mats = {}
    for feature in features:
        feature_data = meta_data[feature]
        sim_type = similarity_type[feature]
        ranking = sim_type == 'jaccard-weighted'
        get_similarity = _sim_func(sim_type)

        feature_mat, feature_lbl = feature2sparse(feature_data, ranking=ranking)
        similarity_mats[feature] = get_similarity(feature_mat)

    return similarity_mats


def combine_distribute_similarity_data(meta_data, similarity_type='jaccard', weights=None):
    # similarity_mats = get_similarity_data(meta_data, similarity_type)
    # iprob = {f: 1+np.prod(s.shape)/s.nnz for f, s in similarity_mats.iteritems()}
    # wsum = np.log(iprob.values()).sum()
    # weights = {f:w/wsum for f, w in iprob.iteritems()}
    raise NotImplementedError


def combine_similarity_data(meta_data, similarity_type='jaccard', weights=None):
    features = meta_data.columns
    num_feat = len(features)

    if isinstance(similarity_type, str):
        similarity_type = [similarity_type] * num_feat

    if not isinstance(similarity_type, dict):
        similarity_type = {k: v for k, v in zip(features, similarity_type)}

    if weights is None:
        weights = [1.0/num_feat] * num_feat

    if not isinstance(weights, dict):
        weights = {k: v for k, v in zip(features, weights)}

    similarity = csc_matrix((meta_data.shape[0],)*2)
    for feature in features:
        feature_data = meta_data[feature]
        sim_type = similarity_type[feature]
        weight = weights[feature]
        ranking = sim_type == 'jaccard-weighted'
        get_similarity = _sim_func(sim_type)

        feature_mat, feature_lbl = feature2sparse(feature_data, ranking=ranking)
        similarity += weight * get_similarity(feature_mat)

    # remove round off errors
    similarity.setdiag(1)
    similarity.data[similarity.data>1] = 1
    return similarity
