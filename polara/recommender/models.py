from polara.recommender import data, defaults
from polara.recommender.evaluation import get_hits, get_relevance_scores, get_ranking_scores
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import svds
from polara.lib.hosvd import tucker_als
from timeit import default_timer as timer



class RecommenderModel(object):
    _config = ('topk', 'filter_seen', 'switch_positive', 'predict_negative')

    def __init__(self, recommender_data):

        self.data = recommender_data
        self._recommendations = None
        self.method = 'Base'
        #if not hasattr(self.data, 'test'):
        #    print 'Submitted data is unprepared. Processing...'
        #    self.data.prepare()

        self._topk = defaults.get_config(['topk'])['topk']
        self.filter_seen  = defaults.get_config(['filter_seen'])['filter_seen']
        self.switch_positive  = defaults.get_config(['switch_positive'])['switch_positive']
        self.predict_negative = defaults.get_config(['predict_negative'])['predict_negative']


    @property
    def recommendations(self):
        if (self._recommendations is None):
            try:
                self._recommendations = self.get_recommendations()
            except AttributeError:
                print '{} model is not ready. Rebuilding.'.format(self.method)
                self.build()
                self._recommendations = self.get_recommendations()
            #print 'recommendations recomputed in {}'.format(self.method)
        return self._recommendations


    @property
    def topk(self):
        return self._topk

    @topk.setter
    def topk(self, new_value):
        #support rolling back scenarion for @k calculations
        if (self._recommendations is not None) and (new_value > self._recommendations.shape[1]):
            self._recommendations = None #if topk is too high - recalculate recommendations
        self._topk = new_value


    def build(self):
        raise NotImplementedError('This must be implemented in subclasses')


    def get_recommendations(self):
        raise NotImplementedError('This must be implemented in subclasses')


    def get_matched_predictions(self):
        userid, itemid = self.data.fields.userid, self.data.fields.itemid
        holdout_data = self.data.test.evalset[itemid]
        holdout = self.data.holdout_size
        holdout_matrix = holdout_data.values.reshape(-1, holdout).astype(np.int64)

        recommendations = self.recommendations #will recalculate if empty

        if recommendations.shape[0] > holdout_matrix.shape[0]:
            print 'Evaluation set is truncated.'
            recommendations = recommendations[:holdout_matrix.shape[0], :]
        elif recommendations.shape[0] < holdout_matrix.shape[0]:
            print 'Recommendations are truncated.'
            holdout_matrix = holdout_matrix[:recommendations.shape[0], :]

        matched_predictions = (recommendations[:, :, None] == holdout_matrix[:, None, :])
        return matched_predictions


    def get_feedback_data(self):
        feedback = self.data.fields.feedback
        eval_data = self.data.test.evalset[feedback].values
        holdout = self.data.holdout_size
        feedback_data = eval_data.reshape(-1, holdout)
        return feedback_data


    def get_positive_feedback(self):
        feedback_data = self.get_feedback_data()
        positive_feedback = feedback_data >= self.switch_positive
        return positive_feedback


    def evaluate(self, method='hits', topk=None):
        #support rolling back scenario for @k calculations
        if topk > self.topk:
            self.topk = topk #will also empty flush old recommendations

        matched_predictions = self.get_matched_predictions()
        matched_predictions = matched_predictions[:, :topk, :]

        if method == 'relevance':
            positive_feedback = self.get_positive_feedback()
            scores = get_relevance_scores(matched_predictions, positive_feedback)
        elif method == 'ranking':
            feedback = self.get_feedback_data()
            scores = get_ranking_scores(matched_predictions, feedback, self.switch_positive)
        elif method == 'hits':
            positive_feedback = self.get_positive_feedback()
            scores = get_hits(matched_predictions, positive_feedback)
        else:
            raise NotImplementedError
        return scores


    @staticmethod
    def downvote_seen_items(recs, idx_seen, min_value=None):
        idx_seen_flat = np.ravel_multi_index(idx_seen, recs.shape)
        min_value = min_value or recs.min()-1
        np.put(recs, idx_seen_flat, min_value)


    def get_topk_items(self, scores):
        topk = self.topk
        recs = np.argsort(scores, axis=1)[:, :-topk-1:-1]
        return recs


    @staticmethod
    def orthogonalize(u, v):
        Qu, Ru = np.linalg.qr(u)
        Qv, Rv = np.linalg.qr(v)
        Ur, Sr, Vr = np.linalg.svd(Ru.dot(Rv.T))
        U = Qu.dot(Ur)
        V = Qv.dot(Vr.T)
        return U, V


class NonPersonalized(RecommenderModel):

    def __init__(self, kind, *args, **kwargs):
        super(NonPersonalized, self).__init__(*args, **kwargs)
        self.method = kind


    def build(self):
        self._recommendations = None


    def get_recommendations(self):
        userid, itemid, feedback = self.data.fields
        test_data = self.data.test.testset
        test_idx = (test_data[userid].values.astype(np.int64),
                    test_data[itemid].values.astype(np.int64))
        num_users = self.data.test.testset[userid].max() + 1

        if self.method == 'mostpopular':
            items_scores = self.data.training.groupby(itemid, sort=True).size().values
            #scores =  np.lib.stride_tricks.as_strided(items_scores, (num_users, items_scores.size), (0, items_scores.itemsize))
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
            #prevent seen items from appearing in recommendations
            self.downvote_seen_items(scores, test_idx)

        top_recs =  self.get_topk_items(scores)
        return top_recs


class CooccurrenceModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(CooccurrenceModel, self).__init__(*args, **kwargs)
        self.method = 'item-to-item' #pick some meaningful name


    def build(self):
        self._recommendations = None
        idx, val, shp = self.data.to_coo(tensor_mode=False)
        #np.ones_like makes feedback implicit
        user_item_matrix = sp.sparse.coo_matrix((np.ones_like(val), (idx[:, 0], idx[:, 1])),
                                          shape=shp, dtype=np.float64).tocsr()

        i2i_matrix = user_item_matrix.T.dot(user_item_matrix)
        #exclude "self-links"
        diag_vals = i2i_matrix.diagonal()
        i2i_matrix -= sp.sparse.dia_matrix((diag_vals, 0), shape=i2i_matrix.shape)
        self._i2i_matrix = i2i_matrix


    def get_recommendations(self):
        userid, itemid, feedback = self.data.fields
        test_data = self.data.test.testset
        i2i_matrix = self._i2i_matrix

        idx = (test_data[userid], test_data[itemid])
        val = np.ones_like(test_data[feedback]) #make feedback implicit
        shp = (idx[0].max()+1, i2i_matrix.shape[0])
        test_matrix = sp.sparse.coo_matrix((val, idx), shape=shp,
                                           dtype=np.float64).tocsr()
        i2i_scores = test_matrix.dot(self._i2i_matrix).A

        if self.filter_seen:
            #prevent seen items from appearing in recommendations
            self.downvote_seen_items(i2i_scores, idx)

        top_recs = self.get_topk_items(i2i_scores)
        return top_recs


class SVDModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(SVDModel, self).__init__(*args, **kwargs)
        self.rank = defaults.svd_rank
        self.method = 'SVD'


    def build(self):
        self._recommendations = None
        idx, val, shp = self.data.to_coo(tensor_mode=False)
        tik = timer()
        svd_matrix = sp.sparse.coo_matrix((val, (idx[:, 0], idx[:, 1])),
                                          shape=shp, dtype=np.float64).tocsr()
        tok = timer() - tik
        print '{} model training time: {}s'.format(self.method, tok)

        _, _, items_factors = svds(svd_matrix, k=self.rank, return_singular_vectors='vh')
        self._items_factors = np.ascontiguousarray(items_factors[::-1, :])


    def get_recommendations(self):
        userid, itemid, feedback = self.data.fields
        test_data = self.data.test.testset

        test_idx = (test_data[userid].values.astype(np.int64),
                    test_data[itemid].values.astype(np.int64))
        test_val = test_data[feedback].values

        v = self._items_factors
        test_shp = (test_data[userid].max()+1,
                    v.shape[1])

        test_matrix = sp.sparse.coo_matrix((test_val, test_idx),
                                           shape=test_shp,
                                           dtype=np.float64).tocsr()

        svd_scores = (test_matrix.dot(v.T)).dot(v)

        if self.predict_negative:
            svd_scores = -svd_scores

        if self.filter_seen:
            #prevent seen items from appearing in recommendations
            self.downvote_seen_items(svd_scores, test_idx)

        top_recs = self.get_topk_items(svd_scores)
        return top_recs


class CoffeeModel(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(CoffeeModel, self).__init__(*args, **kwargs)
        self.mlrank = defaults.mlrank
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
        self._recommendations = None
        idx, val, shp = self.data.to_coo(tensor_mode=True)
        tik = timer()
        users_factors, items_factors, feedback_factors, core = \
                            tucker_als(idx, val, shp, self.mlrank,
                            growth_tol=self.growth_tol,
                            iters = self.num_iters,
                            batch_run=not self.show_output)
        tok = timer() - tik
        print '{} model training time: {}s'.format(self.method, tok)
        self._users_factors = users_factors
        self._items_factors = items_factors
        self._feedback_factors = feedback_factors
        self._core = core


    def get_recommendations(self):
        userid, itemid, feedback = self.data.fields
        v = self._items_factors
        w = self._feedback_factors

        test_shp = (self.data.test.testset[userid].max()+1, v.shape[0], w.shape[0])
        user_idx = self.data.test.testset.loc[:, userid].values.astype(np.int64)
        item_idx = self.data.test.testset.loc[:, itemid].values.astype(np.int64)
        fdbk_idx = self.data.test.testset.loc[:, feedback].values

        fdbk_idx = self.data.index.feedback.set_index('old').loc[fdbk_idx, 'new'].values.astype(np.int64)
        if (fdbk_idx == np.NaN).any():
            raise NotImplementedError('Not all values of feedback are present in training data')

        idx_data = (user_idx, item_idx, fdbk_idx)
        idx_flat = np.ravel_multi_index(idx_data, test_shp)
        shp_flat = (test_shp[0]*test_shp[1], test_shp[2])
        idx = np.unravel_index(idx_flat, shp_flat)

        val = np.ones(self.data.test.testset.shape[0],)
        test_tensor_mat = sp.sparse.coo_matrix((val, idx), shape=shp_flat).tocsr()

        coffee_scores = np.empty((test_shp[0], test_shp[1]))
        chunk = self.chunk
        flattener = self.flattener
        for i in xrange(0, test_shp[0], chunk):
            start = i
            stop = min(i+chunk, test_shp[0])

            test_slice = test_tensor_mat[start*test_shp[1]:stop*test_shp[1], :]
            slice_scores = test_slice.dot(w).reshape(stop-start, test_shp[1], w.shape[1])
            slice_scores = np.tensordot(slice_scores, v, axes=(1, 0))
            slice_scores = np.tensordot(np.tensordot(slice_scores, v, axes=(2, 1)), w, axes=(1, 1))

            coffee_scores[start:stop, :] = self.flatten_scores(slice_scores, flattener)

        if self.filter_seen:
            #prevent seen items from appearing in recommendations
            self.downvote_seen_items(coffee_scores, idx_data[:2])

        top_recs = self.get_topk_items(coffee_scores)
        return top_recs
