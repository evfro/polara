from functools import wraps
from collections import namedtuple
import warnings

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

from polara.recommender import defaults
from polara.recommender.evaluation import get_hits, get_relevance_scores, get_ranking_scores, get_experience_scores
from polara.recommender.evaluation import get_hr_score, get_rr_scores
from polara.recommender.evaluation import assemble_scoring_matrices
from polara.recommender.evaluation import matrix_from_observations
from polara.recommender.utils import array_split
from polara.lib.optimize import simple_pmf_sgd
from polara.lib.tensor import hooi

from polara.preprocessing.matrices import rescale_matrix
from polara.lib.sampler import mf_random_item_scoring, sample_row_wise
from polara.lib.sparse import sparse_dot, inverse_permutation
from polara.lib.sparse import inner_product_at
from polara.lib.sparse import unfold_tensor_coordinates, tensor_outer_at
from polara.tools.random import random_seeds
from polara.tools.timing import track_time

def get_default(name):
    return defaults.get_config([name])[name]


def clean_build_decorator(build_func):
    # this ensures that every time the build function is called,
    # all cached recommendations are cleared
    @wraps(build_func)
    def wrapper(self, *args, **kwargs):
        self.pre_build()
        build_res = build_func(self, *args, **kwargs)
        self.post_build()
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
    def __new__(mcs, name, bases, clsdict):
        cls = super(MetaModel, mcs).__new__(mcs, name, bases, clsdict)
        if 'build' in clsdict:
            setattr(cls, 'build', clean_build_decorator(clsdict['build']))
        return cls


@with_metaclass(MetaModel)
class RecommenderModel(object):
    _config = ('topk', 'filter_seen', 'switch_positive', 'feedback_threshold', 'verify_integrity')
    _pad_const = -1  # used for sparse data

    def __init__(self, recommender_data, feedback_threshold=None):

        self.data = recommender_data
        self._recommendations = None
        self.method = 'ABC'

        self._topk = get_default('topk')
        self._filter_seen = get_default('filter_seen')
        self._feedback_threshold = feedback_threshold or get_default('feedback_threshold')
        self.switch_positive = get_default('switch_positive')
        self.verify_integrity = get_default('verify_integrity')
        self.max_test_workers = get_default('max_test_workers')

        # TODO sorting in data must be by self._prediction_key, also need to change get_test_data
        self._prediction_key = self.data.fields.userid
        self._prediction_target = self.data.fields.itemid

        self._is_ready = False
        self.verbose = True
        self.training_time = [] # setting to None will prevent storing time

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

    @property
    def feedback_threshold(self):
        return self._feedback_threshold

    @feedback_threshold.setter
    def feedback_threshold(self, new_value):
        if self._feedback_threshold != new_value:
            self._feedback_threshold = new_value
            self._renew_model()

    @property
    def filter_seen(self):
        return self._filter_seen

    @filter_seen.setter
    def filter_seen(self, new_value):
        if self._filter_seen != new_value:
            self._filter_seen = new_value
            self._refresh_model()


    def get_base_configuration(self):
        config = {attr: getattr(self, attr) for attr in self._config}
        return config

    def pre_build(self):
        self._is_ready = False

    def build(self):
        raise NotImplementedError('This must be implemented in subclasses')
    
    def post_build(self):
        self._recommendations = None
        self._is_ready = True

    def reuse_model(self):
        raise NotImplementedError('This must be implemented in subclasses')

    def get_training_matrix(self, feedback_threshold=None, ignore_feedback=False,
                            sparse_format='csr', dtype=None):
        threshold = feedback_threshold or self.feedback_threshold
        # the line below also updates data if needed and triggers notifier
        idx, val, shp = self.data.to_coo(tensor_mode=False,
                                         feedback_threshold=threshold)
        dtype = dtype or val.dtype
        if ignore_feedback: # for compatibility with non-numeric tensor feedback data
            val = np.ones_like(val, dtype=dtype)
        matrix = coo_matrix((val, (idx[:, 0], idx[:, 1])),
                            shape=shp, dtype=dtype)
        if sparse_format == 'csr':
            return matrix.tocsr()
        elif sparse_format == 'csc':
            return matrix.tocsc()
        elif sparse_format == 'coo':
            matrix.sum_duplicates()
            return matrix


    def get_test_matrix(self, test_data=None, shape=None, user_slice=None, dtype=None, ignore_feedback=False):
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
        valid_fdbk = fdbk_coo != 0
        if not valid_fdbk.all():
            user_coo = user_coo[valid_fdbk]
            item_coo = item_coo[valid_fdbk]
            fdbk_coo = fdbk_coo[valid_fdbk]

        dtype = dtype or fdbk_coo.dtype
        if ignore_feedback: # for compatibility with non-numeric tensor feedback data
            fdbk_coo = np.ones_like(fdbk_coo, dtype=dtype)

        num_items = shape[1]
        test_matrix = csr_matrix((fdbk_coo, (user_coo, item_coo)),
                                 shape=(num_users, num_items),
                                 dtype=dtype)
        return test_matrix, coo_data


    def _get_slices_idx(self, shape, result_width=None, scores_multiplier=None, dtypes=None):
        result_width = result_width or self.topk
        if scores_multiplier is None:
            try:
                fdbk_dim = self.factors.get(self.data.fields.feedback, None).shape
                scores_multiplier = fdbk_dim[1]
            except AttributeError:
                scores_multiplier = 1

        slices_idx = array_split(shape, result_width, scores_multiplier, dtypes=dtypes)
        return slices_idx


    def _get_test_data(self, feedback_threshold=None):
        try:
            tensor_mode = self.factors.get(self.data.fields.feedback, None) is not None
        except AttributeError:
            tensor_mode = False

        test_shape = self.data.get_test_shape(tensor_mode=tensor_mode)
        threshold = feedback_threshold or self.feedback_threshold
        if self.data.warm_start:
            if threshold:
                print('Specifying threshold has no effect in warm start.')
            threshold = None
        else:
            if self.data.test_sample and (threshold is not None):
                print('Specifying both threshold value and test_sample may change test data.')
        user_idx, item_idx, feedback = self.data.test_to_coo(tensor_mode=tensor_mode, feedback_threshold=threshold)

        idx_diff = np.diff(user_idx)
        # TODO sorting by self._prediction_key
        assert (idx_diff >= 0).all()  # calculations assume testset is sorted by users!

        # TODO only required when testset consists of known users
        if (idx_diff > 1).any() or (user_idx.min() != 0):  # check index monotonicity
            test_users = user_idx[np.r_[0, np.where(idx_diff)[0]+1]]
            user_idx = np.r_[0, np.cumsum(idx_diff > 0)].astype(user_idx.dtype)
        else:
            test_users = np.arange(test_shape[0])

        test_data = (user_idx, item_idx, feedback)

        return test_data, test_shape, test_users


    @staticmethod
    def _slice_test_data(test_data, start, stop):
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
        if not self._is_ready:
            if self.verbose:
                print('{} model is not ready. Rebuilding.'.format(self.method))
            self.build()

        test_data, test_shape, test_users = self._get_test_data()
        if not self.data.warm_start:
            i, = np.where(test_users == i)[0]
        scores, seen_idx = self.slice_recommendations(test_data, test_shape, i, i+1)

        if self.filter_seen:
            self.downvote_seen_items(scores, seen_idx)

        return scores, seen_idx


    def _make_user(self, user_info):
        # converts external user info into internal representation
        userid, itemid, feedback = self.data.fields

        if isinstance(user_info, dict):  # item:feedback dictionary
            items_data, feedback_data = zip(*user_info.items())
        elif isinstance(user_info, (list, tuple, set, np.ndarray)):  # list of items
            items_data = user_info
            feedback_data = {}
            if feedback is not None:
                feedback_val = self.data.training[feedback].max()
                feedback_data = {feedback: [feedback_val]*len(items_data)}
        else:
            raise ValueError("Unrecognized input for `user_info`.")

        try:
            item_index = self.data.index.itemid.training
        except AttributeError:
            item_index = self.data.index.itemid

        # need to convert itemid's to internal representation
        # conversion is not required for feedback (it's made in *to_coo functions, if needed)
        items_data = item_index.set_index('old').loc[items_data, 'new'].values
        user_data = {userid: [0]*len(items_data), itemid: items_data}
        user_data.update(feedback_data)
        return pd.DataFrame(user_data)


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
        if topk is not None:
            self.topk = topk
        try:
            # takes care of both sparse and dense recommendation lists
            top_recs = self.get_topk_elements(scores).squeeze()  # remove singleton
        finally:
            self.topk = _topk

        seen_idx = seen_idx[1]  # only items idx
        # covert back to external representation
        item_index = self.data.get_entity_index(self.data.fields.itemid)
        item_idx_map = item_index.set_index('new')
        top_recs = item_idx_map.loc[top_recs, 'old'].values
        seen_items = item_idx_map.loc[seen_idx, 'old'].values
        return top_recs, seen_items


    def _slice_recommender(self, user_slice, test_data, test_shape, test_users):
        start, stop = user_slice
        scores, slice_data = self.slice_recommendations(test_data, test_shape, start, stop, test_users)
        if self.filter_seen:
            # prevent seen items from appearing in recommendations
            # NOTE: in case of sparse models (e.g. simple item-to-item)
            # there's a risk of having seen items in recommendations list
            # (for topk < i2i_matrix.shape[1]-len(unseen))
            # this is related to low generalization ability
            # of the naive cooccurrence method itself, not to the algorithm
            self.downvote_seen_items(scores, slice_data)
        top_recs = self.get_topk_elements(scores)
        return top_recs


    def run_parallel_recommender(self, result, user_slices, *args):
        with ThreadPoolExecutor(max_workers=self.max_test_workers) as executor:
            recs_futures = {executor.submit(self._slice_recommender,
                                            user_slice, *args): user_slice
                            for user_slice in user_slices}

            for future in as_completed(recs_futures):
                start, stop = recs_futures[future]
                result[start:stop, :] = future.result()


    def run_sequential_recommender(self, result, user_slices, *args):
        for user_slice in user_slices:
            start, stop = user_slice
            result[start:stop, :] = self._slice_recommender(user_slice, *args)


    def get_recommendations(self):
        if self.verify_integrity:
            self.verify_data_integrity()

        test_data = self._get_test_data()
        test_shape = test_data[1]
        user_slices_idx = self._get_slices_idx(test_shape)
        user_slices = zip(user_slices_idx[:-1], user_slices_idx[1:])

        top_recs = np.empty((test_shape[0], self.topk), dtype=np.int64)
        if self.max_test_workers and len(user_slices_idx) > 2:
            self.run_parallel_recommender(top_recs, user_slices, *test_data)
        else:
            self.run_sequential_recommender(top_recs, user_slices, *test_data)
        return top_recs


    def evaluate(self, metric_type='all', topk=None, not_rated_penalty=None,
                 switch_positive=None, ignore_feedback=False, simple_rates=False,
                 on_feedback_level=None):
        if metric_type in ['all', None]:
            metric_type = ['hits', 'relevance', 'ranking', 'experience']

        if metric_type == 'main':
            metric_type = ['relevance', 'ranking']

        if not isinstance(metric_type, (list, tuple)):
            metric_type = [metric_type]

        if int(topk or 0) > self.topk:
            self.topk = topk  # will also flush old recommendations
        # support rolling back scenario for @k calculations
        recommendations = self.recommendations[:, :topk]  # will recalculate if empty
        switch_positive = switch_positive or self.switch_positive
        feedback = self.data.fields.feedback
        holdout = self.data.test.holdout
        if (switch_positive is None) or (feedback is None):
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
            is_positive = (holdout[feedback] >= switch_positive).values

        feedback = None if ignore_feedback else feedback
        scoring_data = assemble_scoring_matrices(recommendations, holdout,
                                                 self._prediction_key, self._prediction_target,
                                                 is_positive, feedback=feedback)

        scores = []
        if 'relevance' in metric_type:  # no need for feedback
            if (self.data.holdout_size == 1) or simple_rates:
                scores.append(get_hr_score(scoring_data[1]))
            else:
                scores.append(get_relevance_scores(*scoring_data, not_rated_penalty=not_rated_penalty))

        if 'ranking' in metric_type:
            if (self.data.holdout_size == 1) or simple_rates:
                scores.append(get_rr_scores(scoring_data[1]))
            else:
                ndcg_alternative = get_default('ndcg_alternative')
                topk = recommendations.shape[1]  # handle topk=None case
                # topk has to be passed explicitly, otherwise it's unclear how to
                # estimate ideal ranking for NDCG and NDCL metrics in get_ndcr_discounts
                # it's also used in MAP calculation
                scores.append(get_ranking_scores(*scoring_data, topk=topk, switch_positive=switch_positive, alternative=ndcg_alternative))

        if 'experience' in metric_type:  # no need for feedback
            fields = self.data.fields
            # support custom scenarios, e.g. coldstart
            entity_type = fields._fields[fields.index(self._prediction_target)]
            entity_index = getattr(self.data.index, entity_type)
            try:
                n_entities = entity_index.shape[0]
            except AttributeError:
                n_entities = entity_index.training.shape[0]
            scores.append(get_experience_scores(recommendations, n_entities))

        if 'hits' in metric_type:  # no need for feedback
            scores.append(get_hits(*scoring_data, not_rated_penalty=not_rated_penalty))

        if not scores:
            raise NotImplementedError

        if len(scores) == 1:
            scores = scores[0]
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
            self.item_scores = item_groups[feedback].sum().values
        else:
            self.item_scores = item_groups.size().values

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        slice_data = self._slice_test_data(test_data, start, stop)
        n_users = stop - start
        scores = np.repeat(self.item_scores[None, :], n_users, axis=0)
        return scores, slice_data


class RandomModel(RecommenderModel):
    def __init__(self, *args, seed=None, **kwargs):
        self.seed = seed
        super(RandomModel, self).__init__(*args, **kwargs)
        self.method = 'RND'

    def build(self):
        pass

    def get_recommendations(self):
        if self.filter_seen:
            test_matrix, _ = self.get_test_matrix()
            n_users, n_items = test_matrix.shape
            assert (test_matrix.getnnz(axis=1) + self.topk <= n_items).all(), "topk value is too large"
            seen_inds = test_matrix.indices
            seen_ptr = test_matrix.indptr
        else: # mimic empty test matrix
            n_users, n_items = self.data.get_test_shape()
            assert self.topk <= n_items, "topk value is too large"
            seen_inds = np.array([], dtype=np.min_scalar_type(n_items))
            seen_ptr = np.broadcast_to(0, n_users+1)
        
        seed_seq = random_seeds(n_users, self.seed)
        recs = sample_row_wise(seen_ptr, seen_inds, n_items, self.topk, seed_seq)
        return recs


class CooccurrenceModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(CooccurrenceModel, self).__init__(*args, **kwargs)
        self.method = 'item-to-item'  # pick some meaningful name
        self.implicit = False
        self.dense_output = False


    def build(self):
        user_item_matrix = self.get_training_matrix()
        if self.implicit:
            # np.sign allows for negative values as well
            user_item_matrix.data = np.sign(user_item_matrix.data)

        with track_time(self.training_time, verbose=self.verbose, model=self.method):
            i2i_matrix = user_item_matrix.T.dot(user_item_matrix)  # gives CSC format
            i2i_matrix.setdiag(0)  # exclude "self-links"
            i2i_matrix.eliminate_zeros()

        self._i2i_matrix = i2i_matrix


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        # NOTE CSR format is mandatory for proper handling of signle user
        # recommendations, as vector of shape (1, N) in CSC format is inefficient

        if self.implicit:
            test_matrix.data = np.sign(test_matrix.data)

        scores = sparse_dot(test_matrix, self._i2i_matrix, self.dense_output, True)
        return scores, slice_data


class ProbabilisticMF(RecommenderModel):
    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', None)
        super().__init__(*args, **kwargs)
        self.method = 'PMF'
        self.optimizer = simple_pmf_sgd
        self.learn_rate = 0.005
        self.sigma = 1
        self.num_epochs = 25
        self.rank = 10
        self.tolerance = 1e-4
        self.factors = {}
        self.rmse_history = None
        self.show_rmse = False
        self.iterations_time = None

    def build(self, *args, **kwargs):
        matrix = self.get_training_matrix(sparse_format='coo', dtype='f8')
        user_idx, item_idx = matrix.nonzero()
        interactions = (user_idx, item_idx, matrix.data)
        nonzero_count = (matrix.getnnz(axis=1), matrix.getnnz(axis=0))
        rank = self.rank
        lrate = self.learn_rate
        sigma = self.sigma
        num_epochs = self.num_epochs
        tol = self.tolerance
        self.rmse_history = []
        self.iterations_time = []

        general_config = dict(seed=self.seed,
                              verbose=self.show_rmse,
                              iter_errors=self.rmse_history,
                              iter_time=self.iterations_time)

        with track_time(self.training_time, verbose=self.verbose, model=self.method):
            P, Q = self.optimizer(interactions, matrix.shape, nonzero_count, rank,
                                  lrate, sigma, num_epochs, tol,
                                  *args,
                                  **kwargs,
                                  **general_config)

        self.factors[self.data.fields.userid] = P
        self.factors[self.data.fields.itemid] = Q

    def get_recommendations(self):
        if self.data.warm_start:
            raise NotImplementedError
        else:
            return super().get_recommendations()


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        slice_data = self._slice_test_data(test_data, start, stop)

        user_factors = self.factors[userid][test_users[start:stop], :]
        item_factors = self.factors[itemid]
        scores = user_factors.dot(item_factors.T)
        return scores, slice_data


class EmbeddingsMixin:
    @property
    def user_embeddings(self):
        return self.factors[self.data.fields.userid]

    @property
    def item_embeddings(self):
        return self.factors[self.data.fields.itemid]


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
                # avoid accidental overwrites if factors backup exists
                self.factors = dict(**self.factors)
                # ellipsis allows to handle 1d array of singular values
                self.factors[entity] = factor[..., :rank]


    def build(self, operator=None, return_factors='vh'):
        if operator is not None:
            svd_matrix = operator
        else:
            svd_matrix = self.get_training_matrix(dtype=np.float64)

        svd_params = dict(k=self.rank, return_singular_vectors=return_factors)

        with track_time(self.training_time, verbose=self.verbose, model=self.method):
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


class ScaledMatrixMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._col_scaling = 0.4
        self._row_scaling = 1
        self.method = f'{self.method}-s'

    @property
    def col_scaling(self):
        return self._col_scaling

    @property
    def row_scaling(self):
        return self._row_scaling

    @col_scaling.setter
    def col_scaling(self, new_value):
        if new_value != self._col_scaling:
            self._col_scaling = new_value
            self._recommendations = None

    @row_scaling.setter
    def row_scaling(self, new_value):
        if new_value != self._row_scaling:
            self._row_scaling = new_value
            self._recommendations = None

    def get_training_matrix(self, *args, **kwargs):
        scaled_matrix = super().get_training_matrix(*args, **kwargs)
        scaled_matrix = rescale_matrix(scaled_matrix, self.row_scaling, 1)
        scaled_matrix = rescale_matrix(scaled_matrix, self.col_scaling, 0)
        return scaled_matrix


class ScaledSVD(ScaledMatrixMixin, SVDModel): pass


class CoffeeModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(CoffeeModel, self).__init__(*args, **kwargs)
        self._mlrank = defaults.mlrank
        self.factors = {}
        self.chunk = defaults.test_chunk_size
        self.method = 'CoFFee'
        self._flattener = defaults.flattener
        self.iter_callback = defaults.iter_callback
        self.growth_tol = defaults.growth_tol
        self.num_iters = defaults.num_iters
        self.show_output = defaults.show_output
        self.seed = None
        self._vectorize_target = defaults.test_vectorize_target
        self.parallel_ttm = defaults.parallel_ttm


    @property
    def mlrank(self):
        return self._mlrank

    @mlrank.setter
    def mlrank(self, new_value):
        if new_value != self._mlrank:
            self._mlrank = new_value
            self._check_reduced_rank(new_value)
            self._recommendations = None

    @property
    def flattener(self):
        return self._flattener

    @flattener.setter
    def flattener(self, new_value):
        old_value = self._flattener
        if new_value != old_value:
            self._flattener = new_value
            self._recommendations = None

    @property
    def tensor_outer_at(self):
        vtarget = self._vectorize_target.lower()
        if self.max_test_workers and (vtarget == 'parallel'):
            # force single thread for tensor_outer_at to safely run in parallel
            vtarget = 'cpu'
        return tensor_outer_at(vtarget)


    def _check_reduced_rank(self, mlrank):
        for mode, entity in enumerate(self.data.fields):
            factor = self.factors.get(entity, None)
            if factor is None:
                continue

            rank = mlrank[mode]
            if factor.shape[1] < rank:
                self._is_ready = False
                self.factors = {}
                break
            elif factor.shape[1] == rank:
                continue
            else:
                # avoid accidental overwrites if factors backup exists
                self.factors = dict(**self.factors)
                rfactor, new_core = self.round_core(self.factors['core'], mode, rank)
                self.factors[entity] = factor.dot(rfactor)
                self.factors['core'] = new_core


    @staticmethod
    def round_core(core, mode, rank):
        new_dims = [mode] + [m for m in range(core.ndim) if m!=mode]
        mode_dim = core.shape[mode]
        flat_core = core.transpose(new_dims).reshape((mode_dim, -1), order='F')
        u, s, vt = np.linalg.svd(flat_core, full_matrices=False)
        rfactor = u[:, :rank]
        new_core = (np.ascontiguousarray(s[:rank, np.newaxis]*vt[:rank, :])
                    .reshape(rank, *[core.shape[i] for i in new_dims[1:]], order='F')
                    .transpose(inverse_permutation(np.array(new_dims))))
        return rfactor, new_core


    @staticmethod
    def flatten_scores(tensor_scores, flattener=None):
        flattener = flattener or slice(None)
        if isinstance(flattener, str):
            slicer = slice(None)
            flatten = getattr(np, flattener)
            matrix_scores = flatten(tensor_scores[..., slicer], axis=-1)
        elif isinstance(flattener, int):
            slicer = flattener
            matrix_scores = tensor_scores[..., slicer]
        elif isinstance(flattener, (list, slice)):
            slicer = flattener
            flatten = np.sum
            matrix_scores = flatten(tensor_scores[..., slicer], axis=-1)
        elif isinstance(flattener, tuple):
            slicer, flatten_method = flattener
            slicer = slicer or slice(None)
            flatten = getattr(np, flatten_method)
            matrix_scores = flatten(tensor_scores[..., slicer], axis=-1)
        elif callable(flattener):
            matrix_scores = flattener(tensor_scores)
        else:
            raise ValueError('Unrecognized value for flattener attribute')
        return matrix_scores


    def build(self):
        idx, val, shp = self.data.to_coo(tensor_mode=True)

        with track_time(self.training_time, verbose=self.verbose, model=self.method):
            *factors, core = hooi(
                idx, val, shp, self.mlrank,
                iter_callback = self.iter_callback,
                growth_tol = self.growth_tol,
                num_iters = self.num_iters,
                verbose = self.show_output,
                parallel_ttm = self.parallel_ttm,
                seed = self.seed
            )
        self.store_factors(core, factors)


    def store_factors(self, core, factors):
        users_factors, items_factors, feedback_factors = factors
        self.factors[self.data.fields.userid] = users_factors
        self.factors[self.data.fields.itemid] = items_factors
        self.factors[self.data.fields.feedback] = feedback_factors
        self.factors['core'] = core


    def reuse_model(self, core, factors):
        self.store_factors(core, factors)
        self.post_build() # signal that the self is ready for use        


    def unfold_test_tensor_slice(self, test_data, shape, start, stop, mode):
        slice_idx = self._slice_test_data(test_data, start, stop)

        num_users = stop - start
        num_items = shape[1]
        num_fdbks = shape[2]
        slice_shp = (num_users, num_items, num_fdbks)

        idx, shp = unfold_tensor_coordinates(slice_idx, slice_shp, mode)
        val = np.ones_like(slice_idx[2], dtype=np.uint8)

        test_tensor_unfolded = csr_matrix((val, idx), shape=shp, dtype=val.dtype)
        return test_tensor_unfolded, slice_idx


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        slice_idx = self._slice_test_data(test_data, start, stop)

        v = self.factors[self.data.fields.itemid]
        w = self.factors[self.data.fields.feedback]

        # use the fact that test data is sorted by users for reduction:
        scores = self.tensor_outer_at(1.0, v, w, slice_idx[1], slice_idx[2])
        scores = np.add.reduceat(scores, np.r_[0, np.where(np.diff(slice_idx[0]))[0]+1])

        wt_flat = self.flatten_scores(w.T, self.flattener) # TODO cache result
        scores = np.tensordot(scores, wt_flat, axes=(2, 0)).dot(v.T)
        return scores, slice_idx

    def get_holdout_slice(self, start, stop):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        holdout = self.data.test.holdout

        user_sel = (holdout[userid] >= start) & (holdout[userid] < stop)
        holdout_users = holdout.loc[user_sel, userid].values.astype(np.int64) - start
        holdout_items = holdout.loc[user_sel, itemid].values.astype(np.int64)
        return (holdout_users, holdout_items)


    # additional functionality: rating pediction
    def predict_feedback(self):
        if self.data.warm_start:
            raise NotImplementedError

        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        feedback = self.data.fields.feedback

        holdout = self.data.test.holdout
        holdout_users = holdout[userid].values.astype(np.int64)
        holdout_items = holdout[itemid].values.astype(np.int64)

        u = self.factors[userid]
        v = self.factors[itemid]
        w = self.factors[feedback]
        g = self.factors['core']

        gv = np.tensordot(g,  v[holdout_items, :], (1, 1))
        gu = (gv * u[holdout_users, None, :].T).sum(axis=0)
        scores = w.dot(gu).T
        predictions = np.argmax(scores, axis=-1)

        feedback_idx = self.data.index.feedback.set_index('new')
        predicted_feedback = feedback_idx.loc[predictions, 'old'].values
        return predicted_feedback


class RandomSampleEvaluationSVDMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # as deifined in RandomSampleEvaluationMixin in data models
        prefix = self.data._holdout_item_prefix
        self._prediction_target = f'{prefix}_{self.data.fields.itemid}'

    def compute_holdout_scores(self, user_factors, item_factors):
        holdout = self.data.test.holdout
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        holdout_size = self.data.holdout_size
        assert holdout_size >= 1 # only fixed number of holdout items is supported

        # "rebase" user index (see comments in `get_recommmendations`)
        useridx = pd.factorize(
            holdout[userid], sort=False  # already sorted via data moodel
        )[0].reshape(-1, holdout_size)
        itemidx = holdout[itemid].values.reshape(-1, holdout_size)
        inner_product = inner_product_at(target='parallel')
        # for general matrix factorization user must take care of submitting
        # user_factors of the correct shape, otherwise, if holdout contains
        # only a subset of all users, the answer will be incorrect
        return inner_product(user_factors, item_factors, useridx, itemidx)

    def compute_random_item_scores(self, user_factors, item_factors):
        holdout = self.data.test.holdout
        userid = self.data.fields.userid
        test_users = holdout[userid].drop_duplicates().values # preserve sorted
        test_items = self.data.unseen_interactions.loc[test_users].values
        # "rebase" user index (see comments in `get_recommmendations`)
        n_users = len(test_users)
        n_items = self.data.unseen_items_num
        useridx = np.broadcast_to(np.arange(n_users)[:, None], (n_users, n_items))
        itemidx = np.concatenate(test_items).reshape(n_users, n_items)
        # perform vectorized scalar products at bulk
        inner_product = inner_product_at(target='parallel')
        # for general matrix factorization user must take care of submitting
        # user_factors of the correct shape, otherwise, if holdout contains
        # only a subset of all users, the answer will be incorrect
        return inner_product(user_factors, item_factors, useridx, itemidx)

    def compute_random_item_scores_gen(self, user_factors, item_factors,
                                       profile_matrix, n_unseen):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        holdout = self.data.test.holdout
        n_users = profile_matrix.shape[0]
        seed = self.data.seed

        holdout_matrix = matrix_from_observations(
            holdout, userid, itemid, profile_matrix.shape, feedback=None
        )
        all_seen = profile_matrix + holdout_matrix # only need nnz indices

        scores = np.zeros((n_users, n_unseen))
        seedseq = np.random.SeedSequence(seed).generate_state(n_users)
        mf_random_item_scoring(
            user_factors, item_factors, all_seen.indptr, all_seen.indices,
            n_unseen, seedseq, scores
        )
        return scores

    def get_recommendations(self):
        itemid = self.data.fields.itemid
        if self._prediction_target == itemid:
            return super().get_recommendations()

        item_factors = self.factors[itemid]
        test_matrix, _ = self.get_test_matrix()
        user_factors = test_matrix.dot(item_factors)
        # from now on will need to work with "rebased" user indices
        # to properly index contiguous user matrices
        holdout_scores = self.compute_holdout_scores(user_factors, item_factors)
        if self.data.unseen_interactions is None:
            n_unseen = self.data.unseen_items_num
            if n_unseen is None:
                raise ValueError('Number of items to sample is unspecified.')

            unseen_scores = self.compute_random_item_scores_gen(
                user_factors, item_factors, test_matrix, n_unseen
            )
        else:
            unseen_scores = self.compute_random_item_scores(
                user_factors, item_factors
            )
        # combine all scores and rank selected items
        scores = np.concatenate((holdout_scores, unseen_scores), axis=1)
        return np.apply_along_axis(self.topsort, 1, scores, self.topk)
