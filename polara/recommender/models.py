# python 2/3 interoperability
from __future__ import print_function
try:
    range = xrange
except NameError:
    pass

from functools import wraps
from collections import namedtuple
import warnings

import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

from polara.recommender import defaults
from polara.recommender.evaluation import get_hits, get_relevance_scores, get_ranking_scores
from polara.recommender.evaluation import get_hr_score, get_mrr_score
from polara.recommender.evaluation import assemble_scoring_matrices
from polara.recommender.utils import array_split, NNZ_MAX
from polara.lib.hosvd import tucker_als
from polara.lib.sparse import csc_matvec
from polara.tools.timing import Timer


def get_default(name):
    return defaults.get_config([name])[name]


def clean_build_decorator(build_func):
    # this ensures that every time the build function is called,
    # all cached recommendations are cleared
    @wraps(build_func)
    def wrapper(self, *args, **kwargs):
        self._is_ready = False
        self._recommendations = None
        build_res = build_func(self, *args, **kwargs)
        self._is_ready = True
        return build_res
    return wrapper


def with_metaclass(mcls):
    # this is used to ensure python 2/3 interoperablity, taken from:
    # https://stackoverflow.com/questions/22409430/portable-meta-class-between-python2-and-python3
    def decorator(cls):
        body = vars(cls).copy()
        # clean out class body
        body.pop('__dict__', None)
        body.pop('__weakref__', None)
        return mcls(cls.__name__, cls.__bases__, body)
    return decorator


class MetaModel(type):
    # performs cleaning of the instance when build method is called
    # propagates the action to any subclasses, key idea is borrowed from here:
    # https://stackoverflow.com/questions/18858759/python-decorating-a-class-method-that-is-intended-to-be-overwritten-when-inheri
    def __new__(meta, name, bases, clsdict):
        cls = super(MetaModel, meta).__new__(meta, name, bases, clsdict)
        if 'build' in clsdict:
            setattr(cls, 'build', clean_build_decorator(clsdict['build']))
        return cls


@with_metaclass(MetaModel)
class RecommenderModel(object):
    _config = ('topk', 'filter_seen', 'switch_positive', 'verify_integrity')
    _pad_const = -1  # used for sparse data

    def __init__(self, recommender_data, switch_positive=None):

        self.data = recommender_data
        self._recommendations = None
        self.method = 'ABC'

        self._topk = get_default('topk')
        self.filter_seen = get_default('filter_seen')
        # `switch_positive` can be used by other models during construction process
        # (e.g. mymedialite wrapper or any other implicit model); hence, it's
        # better to make it a model attribute, not a simple evaluation argument
        # (in contrast to `on_feedback_level` argument of self.evaluate)
        self.switch_positive = switch_positive or get_default('switch_positive')
        self.verify_integrity = get_default('verify_integrity')

        # TODO sorting in data must be by self._key, also need to change get_test_data
        self._key = self.data.fields.userid
        self._target = self.data.fields.itemid

        self._is_ready = False
        self.verbose = True

        self.data.subscribe(self.data.on_change_event, self._renew_model)
        self.data.subscribe(self.data.on_update_event, self._refresh_model)


    @property
    def recommendations(self):
        if self._recommendations is None:
            if not self._is_ready:
                if self.verbose:
                    print('{} model is not ready. Rebuilding.'.format(self.method))
                self.build()
            self._recommendations = self.get_recommendations()
        return self._recommendations


    def _renew_model(self):
        self._recommendations = None
        self._is_ready = False

    def _refresh_model(self):
        self._recommendations = None


    @property
    def topk(self):
        return self._topk

    @topk.setter
    def topk(self, new_value):
        # support rolling back scenarion for @k calculations
        if (self._recommendations is not None) and (new_value > self._recommendations.shape[1]):
            self._recommendations = None  # if topk is too high - recalculate recommendations
        self._topk = new_value


    def build(self):
        raise NotImplementedError('This must be implemented in subclasses')


    def get_training_matrix(self, dtype=None):
        idx, val, shp = self.data.to_coo(tensor_mode=False)
        dtype = dtype or val.dtype
        matrix = csr_matrix((val, (idx[:, 0], idx[:, 1])),
                            shape=shp, dtype=dtype)
        return matrix


    def get_test_matrix(self, test_data=None, shape=None, user_slice=None):
        if test_data is None:
            test_data, shape, _ = self._get_test_data()
        elif shape is None:
            raise ValueError('Shape of test data must be provided')

        num_users_all = shape[0]
        if user_slice:
            start, stop = user_slice
            stop = min(stop, num_users_all)
            num_users = stop - start
            coo_data = self._slice_test_data(test_data, start, stop)
        else:
            num_users = num_users_all
            coo_data = test_data

        user_coo, item_coo, fdbk_coo = coo_data
        num_items = shape[1]
        test_matrix = csr_matrix((fdbk_coo, (user_coo, item_coo)),
                                 shape=(num_users, num_items),
                                 dtype=fdbk_coo.dtype)
        return test_matrix, coo_data


    def _get_slices_idx(self, shape, result_width=None, scores_multiplier=None, dtypes=None):
        result_width = result_width or self.topk
        if scores_multiplier is None:
            try:
                fdbk_dim = self.factors.get(self.data.fields.feedback, None).shape
                scores_multiplier = fdbk_dim[0] + 2*fdbk_dim[1]
            except AttributeError:
                scores_multiplier = 1

        slices_idx = array_split(shape, result_width, scores_multiplier, dtypes=dtypes)
        return slices_idx


    def _get_test_data(self):
        try:
            tensor_mode = self.factors.get(self.data.fields.feedback, None) is not None
        except AttributeError:
            tensor_mode = False

        user_idx, item_idx, feedback = self.data.test_to_coo(tensor_mode=tensor_mode)
        test_shape = self.data.get_test_shape(tensor_mode=tensor_mode)

        idx_diff = np.diff(user_idx)
        # TODO sorting by self._key
        assert (idx_diff >= 0).all()  # calculations assume testset is sorted by users!

        # TODO only required when testset consists of known users
        if (idx_diff > 1).any() or (user_idx.min() != 0):  # check index monotonicity
            test_users = user_idx[np.r_[0, np.where(idx_diff)[0]+1]]
            user_idx = np.r_[0, np.cumsum(idx_diff > 0)].astype(user_idx.dtype)
        else:
            test_users = np.arange(test_shape[0])

        test_data = (user_idx, item_idx, feedback)

        return test_data, test_shape, test_users


    def _slice_test_data(self, test_data, start, stop):
        user_coo, item_coo, fdbk_coo = test_data

        slicer = (user_coo >= start) & (user_coo < stop)
        # always slice over users only
        user_slice_coo = user_coo[slicer] - start
        item_slice_coo = item_coo[slicer]
        fdbk_slice_coo = fdbk_coo[slicer]

        return (user_slice_coo, item_slice_coo, fdbk_slice_coo)


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        raise NotImplementedError('This must be implemented in subclasses')


    def _user_scores(self, i):
        # should not be exposed, designed for use within framework
        # operates on internal itemid's
        test_data, test_shape, _ = self._get_test_data()
        scores, seen_idx = self.slice_recommendations(test_data, test_shape, i, i+1)

        if self.filter_seen:
            self.downvote_seen_items(scores, seen_idx)

        return scores, seen_idx


    def _make_user(self, user_info):
        # converts external user info into internal representation
        userid, itemid, feedback = self.data.fields

        if isinstance(user_info, dict):
            user_info = user_info.items()
        try:
            items_data, feedback_data = zip(*user_info)
        except TypeError:
            items_data = user_info
            feedback_val = self.data.training[feedback].max()
            feedback_data = [feedback_val]*len(items_data)

        try:
            item_index = self.data.index.itemid.training
        except AttributeError:
            item_index = self.data.index.itemid

        # need to convert itemid's to internal representation
        # conversion is not required for feedback (it's made in *to_coo functions, if needed)
        items_data = item_index.set_index('old').loc[items_data, 'new'].values
        user_data = pd.DataFrame({userid: [0]*len(items_data),
                                 itemid: items_data,
                                 feedback: feedback_data})
        return user_data


    def show_recommendations(self, user_info, topk=None):
        # convenience function to model users and get recs
        # operates on external itemid's
        if isinstance(user_info, int):
            scores, seen_idx = self._user_scores(user_info)
        else:
            testset = self.data.test.testset
            holdout = self.data.test.holdout
            user_data = self._make_user(user_info)
            try:
                # makes a "fake" test user
                self.data._test = namedtuple('TestData', 'testset holdout')._make([user_data, None])
                scores, seen_idx = self._user_scores(0)
            finally:
                # restore original data - prevent information loss
                self.data._test = namedtuple('TestData', 'testset holdout')._make([testset, holdout])

        _topk = self.topk
        self.topk = topk or _topk
        # takes care of both sparse and dense recommendation lists
        top_recs = self.get_topk_elements(scores).squeeze()  # remove singleton
        self.topk = _topk

        try:
            item_index = self.data.index.itemid.training
        except AttributeError:
            item_index = self.data.index.itemid

        seen_idx = seen_idx[1]  # only items idx
        # covert back to external representation
        item_idx_map = item_index.set_index('new')
        top_recs = item_idx_map.loc[top_recs, 'old'].values
        seen_items = item_idx_map.loc[seen_idx, 'old'].values
        return top_recs, seen_items


    def get_recommendations(self):
        if self.verify_integrity:
            self.verify_data_integrity()

        test_data, test_shape, test_users = self._get_test_data()

        topk = self.topk
        top_recs = np.empty((test_shape[0], topk), dtype=np.int64)

        user_slices = self._get_slices_idx(test_shape)
        start = user_slices[0]
        for i in user_slices[1:]:
            stop = i
            scores, slice_data = self.slice_recommendations(test_data, test_shape, start, stop, test_users)

            if self.filter_seen:
                # prevent seen items from appearing in recommendations
                # NOTE: in case of sparse models (e.g. simple item-to-item)
                # there's a risk of having seen items in recommendations list
                # (for topk < i2i_matrix.shape[1]-len(unseen))
                # this is related to low generalization ability
                # of the naive cooccurrence method itself, not to the algorithm
                self.downvote_seen_items(scores, slice_data)

            top_recs[start:stop, :] = self.get_topk_elements(scores)
            start = stop

        return top_recs


    def evaluate(self, method='hits', topk=None, not_rated_penalty=None, on_feedback_level=None):
        feedback = self.data.fields.feedback
        if int(topk or 0) > self.topk:
            self.topk = topk  # will also flush old recommendations

        # support rolling back scenario for @k calculations
        recommendations = self.recommendations[:, :topk]  # will recalculate if empty

        eval_data = self.data.test.holdout
        if self.switch_positive is None:
            # all recommendations are considered positive predictions
            # this is a proper setting for binary data problems (implicit feedback)
            # in this case all unrated items, recommended by an algorithm
            # assumed to be "honest" false positives and therefore penalty equals 1
            not_rated_penalty = 1 if not_rated_penalty is None else not_rated_penalty
            is_positive = None
        else:
            # if data is not binary (explicit feedback), the intuition is different
            # it becomes unclear whether unrated items are "honest" false positives
            # as among these items can be both top rated and down-rated
            # the defualt setting in this case is to ignore such items at all
            # by setting penalty to 0, however, it is adjustable
            not_rated_penalty = not_rated_penalty or 0
            is_positive = (eval_data[feedback] >= self.switch_positive).values

        scoring_data = assemble_scoring_matrices(recommendations, eval_data,
                                                 self._key, self._target,
                                                 is_positive, feedback=feedback)

        if method == 'relevance':  # no need for feedback
            if self.data.holdout_size == 1:
                scores = get_hr_score(scoring_data[1])
            else:
                scores = get_relevance_scores(*scoring_data, not_rated_penalty=not_rated_penalty)
        elif method == 'ranking':
            if self.data.holdout_size == 1:
                scores = get_mrr_score(scoring_data[1])
            else:
                ndcg_alternative = get_default('ndcg_alternative')
                topk = recommendations.shape[1]  # handle topk=None case
                # topk has to be passed explicitly, otherwise it's unclear how to
                # estimate ideal ranking for NDCG and NDCL metrics in get_ndcr_discounts
                scores = get_ranking_scores(*scoring_data, switch_positive=self.switch_positive, topk=topk, alternative=ndcg_alternative)
        elif method == 'hits':  # no need for feedback
            scores = get_hits(*scoring_data, not_rated_penalty=not_rated_penalty)
        else:
            raise NotImplementedError
        return scores


    @staticmethod
    def topsort(a, topk):
        parted = np.argpartition(a, -topk)[-topk:]
        return parted[np.argsort(-a[parted])]


    @staticmethod
    def downvote_seen_items(recs, idx_seen):
        # NOTE for sparse scores matrix this method can lead to a slightly worse
        # results (comparing to the same method but with "densified" scores matrix)
        # models with sparse scores can alleviate that by extending recommendations
        # list with most popular items or items generated by a more sophisticated logic
        idx_seen = idx_seen[:2]  # need only users and items
        if sp.sparse.issparse(recs):
            ind_data = np.ones(len(idx_seen[0]), dtype=np.bool)  # indicator
            seen = coo_matrix((ind_data, idx_seen), shape=recs.shape, copy=False)
            seen_recs = recs.multiply(seen)
            # In the sparse case it's impossible to downvote seen items scores
            # without making matrix dense. Have to simply make them 0.
            recs -= seen_recs
            # This, however, differs from the dense case results as seen
            # items may appear earlier in the top-k list due to randomization
        else:
            try:
                idx_seen_flat = np.ravel_multi_index(idx_seen, recs.shape)
            except ValueError:
                # make compatible for single user recommendations
                idx_seen_flat = idx_seen
            seen_data = recs.flat[idx_seen_flat]
            # move seen items scores below minimum value
            lowered = recs.min() - (seen_data.max() - seen_data) - 1
            recs.flat[idx_seen_flat] = lowered


    def get_topk_elements(self, scores):
        topk = self.topk
        if sp.sparse.issparse(scores):
            assert scores.format == 'csr'
            # there can be less then topk values in some rows
            # need to extend sorted scores to conform with evaluation matrix shape
            # can do this by adding -1's to the right, however:
            # this relies on the fact that there are no -1's in evaluation matrix
            # NOTE need to ensure that this is always true

            def topscore(x, k):
                data = x.data.values
                cols = x.cols.values
                nnz = len(data)
                if k >= nnz:
                    cols_sorted = cols[np.argsort(-data)]
                    # need to pad values to conform with evaluation matrix shape
                    res = np.pad(cols_sorted, (0, k-nnz),
                                 'constant', constant_values=self._pad_const)
                else:
                    # TODO verify, that even if k is relatively small, then
                    # argpartition doesn't add too much overhead?
                    res = cols[self.topsort(data, k)]
                return res

            idx = scores.nonzero()
            row_data = pd.DataFrame({'data': scores.data, 'cols': idx[1]}).groupby(idx[0], sort=True)
            nnz_users = row_data.grouper.levels[0]
            num_users = scores.shape[0]
            if len(nnz_users) < num_users:
                # scores may have zero-valued rows, this breaks get_topk_elements
                # as scores.nonzero() will filter out indices of those rows.
                # Need to restore full data with zeros in that case.
                recs = np.empty((num_users, topk), dtype=idx[1].dtype)
                zero_rows = np.in1d(np.arange(num_users), nnz_users, assume_unique=True, invert=True)
                recs[zero_rows, :] = self._pad_const
                recs[~zero_rows, :] = np.stack(row_data.apply(topscore, topk).tolist())
            else:
                recs = np.stack(row_data.apply(topscore, topk).tolist())
        else:
            # apply_along_axis is more memory efficient then argsort on full array
            recs = np.apply_along_axis(self.topsort, 1, scores, topk)
        return recs


    @staticmethod
    def orthogonalize(u, v, complete=False):
        Qu, Ru = np.linalg.qr(u)
        Qv, Rv = np.linalg.qr(v)
        if complete:
            # it's not needed for folding-in, as Ur and Vr will cancel out anyway
            Ur, Sr, Vr = np.linalg.svd(Ru.dot(Rv.T))
            U = Qu.dot(Ur)
            V = Qv.dot(Vr.T)
        else:
            U, V = Qu, Qv
        return U, V


    def verify_data_integrity(self):
        data = self.data
        userid, itemid, feedback = data.fields

        try:
            item_index = data.index.itemid.training
        except AttributeError:
            item_index = data.index.itemid

        nunique_items = data.training[itemid].nunique()
        assert nunique_items == item_index.shape[0]
        assert nunique_items == data.training[itemid].max() + 1

        testset = data.test.testset
        if testset is not None:
            nunique_test_users = testset[userid].nunique()
            if data._state == 4:
                assert nunique_test_users == testset[userid].max() + 1

        try:
            assert self.factors.get(itemid, None).shape[0] == item_index.shape[0]
            assert self.factors.get(feedback, None).shape[0] == data.index.feedback.shape[0]
        except AttributeError:
            pass


class NonPersonalized(RecommenderModel):

    def __init__(self, kind, *args, **kwargs):
        deprecation_msg = '''This is a deprecated method.
        Use either PopularityModel or RandomModel instead.'''
        warnings.warn(deprecation_msg, DeprecationWarning)
        super(NonPersonalized, self).__init__(*args, **kwargs)
        self.method = kind


    def build(self):
        pass


    def get_recommendations(self):
        userid, itemid, feedback = self.data.fields
        test_data = self.data.test.testset
        test_idx = (test_data[userid].values.astype(np.int64),
                    test_data[itemid].values.astype(np.int64))
        num_users = self.data.test.testset[userid].max() + 1

        if self.method == 'mostpopular':
            items_scores = self.data.training.groupby(itemid, sort=True).size().values
            # scores =  np.lib.stride_tricks.as_strided(items_scores, (num_users, items_scores.size), (0, items_scores.itemsize))
            scores = np.repeat(items_scores[None, :], num_users, axis=0)
        elif self.method == 'random':
            num_items = self.data.training[itemid].max() + 1
            scores = np.random.random((num_users, num_items))
        elif self.method == 'topscore':
            items_scores = self.data.training.groupby(itemid, sort=True)[feedback].sum().values
            scores = np.repeat(items_scores[None, :], num_users, axis=0)
        else:
            raise NotImplementedError

        if self.filter_seen:
            # prevent seen items from appearing in recommendations
            self.downvote_seen_items(scores, test_idx)

        top_recs = self.get_topk_elements(scores)
        return top_recs


class PopularityModel(RecommenderModel):
    def __init__(self, *args, **kwargs):
        super(PopularityModel, self).__init__(*args, **kwargs)
        self.method = 'MP'
        self.by_feedback_value = False

    def build(self):
        itemid = self.data.fields.itemid
        item_groups = self.data.training.groupby(itemid, sort=True)
        if self.by_feedback_value:
            feedback = self.data.fields.feedback
            self.items_scores = item_groups[feedback].sum().values
        else:
            self.item_scores = item_groups.size().values

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        slice_data = self._slice_test_data(test_data, start, stop)
        n_users = stop - start
        scores = np.repeat(self.item_scores[None, :], n_users, axis=0)
        return scores, slice_data


class RandomModel(RecommenderModel):
    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', None)
        super(RandomModel, self).__init__(*args, **kwargs)
        self.method = 'RND'

    def build(self):
        try:
            index_data = self.data.index.itemid.training
        except AttributeError:
            index_data = self.data.index.itemid
        self.n_items = index_data.shape[0]
        self._random_state = np.random.RandomState(self.seed) if self.seed else np.random

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        slice_data = self._slice_test_data(test_data, start, stop)
        n_users = stop - start
        scores = self._random_state.rand(n_users, self.n_items)
        return scores, slice_data


class CooccurrenceModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(CooccurrenceModel, self).__init__(*args, **kwargs)
        self.method = 'item-to-item'  # pick some meaningful name
        self.implicit = True
        self.dense_output = False


    def build(self):
        user_item_matrix = self.get_training_matrix()
        if self.implicit:
            # np.sign allows for negative values as well
            user_item_matrix.data = np.sign(user_item_matrix.data)

        with Timer(self.method, verbose=self.verbose):
            i2i_matrix = user_item_matrix.T.dot(user_item_matrix)  # gives CSC format
            i2i_matrix.setdiag(0)  # exclude "self-links"
            i2i_matrix.eliminate_zeros()

        self._i2i_matrix = i2i_matrix


    def _sparse_dot(self, tst_mat, i2i_mat):
        # scipy always returns sparse result, even if dot product is dense
        # this function offers solution to this problem
        # it also takes care on sparse result w.r.t. to further processing
        if self.dense_output:  # calculate dense result directly
            # TODO matmat multiplication instead of iteration with matvec
            res_type = np.result_type(i2i_mat.dtype, tst_mat.dtype)
            scores = np.empty((tst_mat.shape[0], i2i_mat.shape[1]), dtype=res_type)
            for i in range(tst_mat.shape[0]):
                v = tst_mat.getrow(i)
                scores[i, :] = csc_matvec(i2i_mat, v, dense_output=True, dtype=res_type)
        else:
            scores = tst_mat.dot(i2i_mat.T)
            # NOTE even though not neccessary for symmetric i2i matrix,
            # transpose helps to avoid expensive conversion to CSR (performed by scipy)
            if scores.nnz > NNZ_MAX:
                # too many nnz lead to undesired memory overhead in downvote_seen_items
                scores = scores.toarray(order='C')
        return scores


    def slice_recommendations(self, test_data, shape, start, stop, test_user=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        # NOTE CSR format is mandatory for proper handling of signle user
        # recommendations, as vector of shape (1, N) in CSC format is inefficient

        if self.implicit:
            test_matrix.data = np.sign(test_matrix.data)

        scores = self._sparse_dot(test_matrix, self._i2i_matrix)
        return scores, slice_data


class SVDModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(SVDModel, self).__init__(*args, **kwargs)
        self._rank = defaults.svd_rank
        self.method = 'PureSVD'
        self.factors = {}

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, new_value):
        if new_value != self._rank:
            self._rank = new_value
            self._check_reduced_rank(new_value)
            self._recommendations = None

    def _check_reduced_rank(self, rank):
        for entity, factor in self.factors.items():
            if factor is None:
                continue

            if factor.shape[-1] < rank:
                self._is_ready = False
                self.factors = dict.fromkeys(self.factors.keys())
                break
            else:
                self.factors[entity] = factor[..., :rank]


    def build(self, operator=None, return_factors='vh'):
        if operator is not None:
            svd_matrix = operator
        else:
            svd_matrix = self.get_training_matrix(dtype=np.float64)

        svd_params = dict(k=self.rank, return_singular_vectors=return_factors)

        with Timer(self.method, verbose=self.verbose):
            user_factors, sigma, item_factors = svds(svd_matrix, **svd_params)

        if user_factors is not None:
            user_factors = np.ascontiguousarray(user_factors[:, ::-1])
        if item_factors is not None:
            item_factors = np.ascontiguousarray(item_factors[::-1, :]).T
        if sigma is not None:
            sigma = np.ascontiguousarray(sigma[::-1])

        self.factors[self.data.fields.userid] = user_factors
        self.factors[self.data.fields.itemid] = item_factors
        self.factors['singular_values'] = sigma

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        v = self.factors[self.data.fields.itemid]
        scores = (test_matrix.dot(v)).dot(v.T)
        return scores, slice_data


class CoffeeModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(CoffeeModel, self).__init__(*args, **kwargs)
        self.mlrank = defaults.mlrank
        self.factors = {}
        self.chunk = defaults.test_chunk_size
        self.method = 'CoFFee'
        self._flattener = defaults.flattener
        self.growth_tol = defaults.growth_tol
        self.num_iters = defaults.num_iters
        self.show_output = defaults.show_output


    @property
    def flattener(self):
        return self._flattener

    @flattener.setter
    def flattener(self, new_value):
        old_value = self._flattener
        if new_value != old_value:
            self._flattener = new_value
            self._recommendations = None


    @staticmethod
    def flatten_scores(tensor_scores, flattener=None):
        flattener = flattener or slice(None)
        if isinstance(flattener, str):
            slicer = slice(None)
            flatten = getattr(np, flattener)
            matrix_scores = flatten(tensor_scores[:, :, slicer], axis=-1)
        elif isinstance(flattener, int):
            slicer = flattener
            matrix_scores = tensor_scores[:, :, slicer]
        elif isinstance(flattener, list) or isinstance(flattener, slice):
            slicer = flattener
            flatten = np.sum
            matrix_scores = flatten(tensor_scores[:, :, slicer], axis=-1)
        elif isinstance(flattener, tuple):
            slicer, flatten_method = flattener
            slicer = slicer or slice(None)
            flatten = getattr(np, flatten_method)
            matrix_scores = flatten(tensor_scores[:, :, slicer], axis=-1)
        elif callable(flattener):
            matrix_scores = flattener(tensor_scores)
        else:
            raise ValueError('Unrecognized value for flattener attribute')
        return matrix_scores


    def build(self):
        idx, val, shp = self.data.to_coo(tensor_mode=True)

        with Timer(self.method, verbose=self.verbose):
            users_factors, items_factors, feedback_factors, core = \
                tucker_als(idx, val, shp, self.mlrank,
                           growth_tol=self.growth_tol,
                           iters=self.num_iters,
                           batch_run=not self.show_output)

        self.factors[self.data.fields.userid] = users_factors
        self.factors[self.data.fields.itemid] = items_factors
        self.factors[self.data.fields.feedback] = feedback_factors
        self.factors['core'] = core


    def get_test_tensor(self, test_data, shape, start, end):
        slice_idx = self._slice_test_data(test_data, start, end)

        num_users = end - start
        num_items = shape[1]
        num_fdbks = shape[2]
        slice_shp = (num_users, num_items, num_fdbks)

        idx_flat = np.ravel_multi_index(slice_idx, slice_shp)
        shp_flat = (num_users*num_items, num_fdbks)
        idx = np.unravel_index(idx_flat, shp_flat)
        val = np.ones_like(slice_idx[2])

        test_tensor_unfolded = csr_matrix((val, idx), shape=shp_flat, dtype=val.dtype)
        return test_tensor_unfolded, slice_idx


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_tensor_unfolded, slice_idx = self.get_test_tensor(test_data, shape, start, stop)
        v = self.factors[self.data.fields.itemid]
        w = self.factors[self.data.fields.feedback]

        num_users = stop - start
        num_items = shape[1]
        num_fdbks = shape[2]

        # assume that w.shape[1] < v.shape[1] (allows for more efficient calculations)
        scores = test_tensor_unfolded.dot(w).reshape(num_users, num_items, num_fdbks)
        scores = np.tensordot(scores, v, axes=(1, 0))
        scores = np.tensordot(np.tensordot(scores, v, axes=(2, 1)), w, axes=(1, 1))
        scores = self.flatten_scores(scores, self.flattener)
        return scores, slice_idx

    # additional functionality: rating pediction
    def get_holdout_slice(self, start, stop):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        eval_data = self.data.test.holdout

        user_sel = (eval_data[userid] >= start) & (eval_data[userid] < stop)
        holdout_users = eval_data.loc[user_sel, userid].values.astype(np.int64) - start
        holdout_items = eval_data.loc[user_sel, itemid].values.astype(np.int64)
        return (holdout_users, holdout_items)


    def predict_feedback(self):
        flattener_old = self.flattener
        self.flattener = 'argmax'  # this will be applied along feedback axis
        feedback_idx = self.data.index.feedback.set_index('new')

        test_data, test_shape, _ = self._get_test_data()
        holdout_size = self.data.holdout_size
        dtype = feedback_idx.old.dtype
        predicted_feedback = np.empty((test_shape[0], holdout_size), dtype=dtype)

        user_slices = self._get_slices_idx(test_shape, result_width=holdout_size)
        start = user_slices[0]
        for i in user_slices[1:]:
            stop = i
            predicted, _ = self.slice_recommendations(test_data, test_shape, start, stop)
            holdout_idx = self.get_holdout_slice(start, stop)
            feedback_values = feedback_idx.loc[predicted[holdout_idx], 'old'].values
            predicted_feedback[start:stop, :] = feedback_values.reshape(-1, holdout_size)
            start = stop
        self.flattener = flattener_old
        return predicted_feedback
