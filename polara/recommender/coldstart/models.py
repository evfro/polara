import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy as sp

from polara import SVDModel
from polara.recommender.models import RecommenderModel, ScaledMatrixMixin
from polara.recommender.hybrid.models import LCEModel, HybridSVD
from polara.recommender.external.lightfm.lightfmwrapper import LightFMWrapper
from polara.lib.similarity import stack_features
from polara.lib.sparse import sparse_dot


class ItemColdStartEvaluationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_seen = False # there are no seen entities in cold start
        self._prediction_key = '{}_cold'.format(self.data.fields.itemid)
        self._prediction_target = self.data.fields.userid


class ItemColdStartRecommenderMixin:
    def get_recommendations(self):
        if self.verify_integrity:
            self.verify_data_integrity()

        cold_item_meta = self.item_features.reindex(
            self.data.index.itemid.cold_start.old.values,
            fill_value=[]
        )

        n_test_items = cold_item_meta.shape[0]
        try:
            n_test_users = self.data.representative_users.shape[0]
        except AttributeError:
            n_test_users = self.data.index.userid.training.shape[0]

        test_shape = (n_test_items, n_test_users)
        cold_slices_idx = self._get_slices_idx(test_shape)
        cold_slices = zip(cold_slices_idx[:-1], cold_slices_idx[1:])

        result = np.empty((test_shape[0], self.topk), dtype=np.int64)
        if self.max_test_workers and len(cold_slices_idx) > 2:
            self.run_parallel_recommender(result, cold_slices, cold_item_meta)
        else:
            self.run_sequential_recommender(result, cold_slices, cold_item_meta)
        return result

    def _slice_recommender(self, cold_slice, cold_item_meta):
        start, stop = cold_slice
        scores = self.slice_recommendations(cold_item_meta, start, stop)
        top_recs = self.get_topk_elements(scores)
        return top_recs


class RandomModelItemColdStart(ItemColdStartEvaluationMixin, RecommenderModel):
    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', None)
        super().__init__(*args, **kwargs)
        self.method = 'RND(cs)'

    def build(self):
        seed = self.seed
        self._random_state = np.random.RandomState(seed) if seed is not None else np.random

    def get_recommendations(self):
        repr_users = self.data.representative_users
        if repr_users is None:
            repr_users = self.data.index.userid.training
        repr_users = repr_users.new.values
        n_cold_items = self.data.index.itemid.cold_start.shape[0]
        shape = (n_cold_items, len(repr_users))
        users_matrix = np.lib.stride_tricks.as_strided(repr_users, shape,
                                                       (0, repr_users.itemsize))
        random_users = np.apply_along_axis(self._random_state.choice, 1,
                                           users_matrix, self.topk, replace=False)
        return random_users


class PopularityModelItemColdStart(ItemColdStartEvaluationMixin, RecommenderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'MP(cs)'

    def build(self):
        userid = self.data.fields.userid
        user_activity = self.data.training[userid].value_counts(sort=False)
        repr_users = self.data.representative_users
        if repr_users is not None:
            user_activity = user_activity.reindex(repr_users.new.values)
        self.user_scores = user_activity.sort_values(ascending=False)

    def get_recommendations(self):
        topk = self.topk
        shape = (self.data.index.itemid.cold_start.shape[0], topk)
        top_users = self.user_scores.index[:topk].values
        top_users_array = np.lib.stride_tricks.as_strided(top_users, shape,
                                                          (0, top_users.itemsize))
        return top_users_array


class SimilarityAggregationItemColdStart(ItemColdStartEvaluationMixin, RecommenderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'SIM(cs)'
        self.implicit = False
        self.dense_output = False

    def build(self):
        pass

    def get_recommendations(self):
        item_similarity_scores = self.data.cold_items_similarity

        user_item_matrix = self.get_training_matrix()
        if self.implicit:
            user_item_matrix.data = np.ones_like(user_item_matrix.data)
        scores = sparse_dot(item_similarity_scores, user_item_matrix, self.dense_output, True)
        top_similar_users = self.get_topk_elements(scores).astype(np.intp)
        return top_similar_users


class LCEModelItemColdStart(ItemColdStartEvaluationMixin, ItemColdStartRecommenderMixin, LCEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'LCE(cs)'
        self.item_features_invgram = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        Hs = self.factors['item_features'].T
        self.item_features_invgram = np.linalg.pinv(Hs @ Hs.T)

    def slice_recommendations(self, cold_item_meta, start, stop):
        cold_slice_meta = cold_item_meta.iloc[start:stop]
        cold_item_features, _ = stack_features(
            cold_slice_meta,
            labels=self.item_features_labels,
            normalize=False)

        Hu = self.factors[self.data.fields.userid].T
        Hs = self.factors['item_features'].T

        cold_items_factors = cold_item_features.dot(Hs.T).dot(self.item_features_invgram)
        cold_items_factors[cold_items_factors < 0] = 0
        scores = cold_items_factors @ Hu
        return scores


class ItemColdStartSVDModelMixin:
    def __init__(self, *args, item_features=None, **kwargs):
        super().__init__(*args, **kwargs)
        if item_features is None: # assume features are provided via data model
            item_features = self.data.item_features
        assert item_features is not None
        self.item_features = item_features
        self.item_features_labels = None
        self._item_features_transform_helper = None
        self.data.subscribe(self.data.on_change_event, self._clean_metadata)

    def _clean_metadata(self):
        self.item_features_labels = None

    @property
    def item_features_embeddings(self):
        itemid = self.data.fields.itemid
        item_features_key = f'{itemid}_features'
        return self.factors.get(item_features_key, None)

    def _round_item_features_transform(self):
        try:
            rank = self.item_features_embeddings.shape[1]
        except AttributeError: # embeddings are None (not computed yet)
            self._item_features_transform_helper = None
        else:
            transform_rank = self._item_features_transform_helper.shape[0]
            if transform_rank > rank: # round transform
                self.update_item_features_transform()
            else:
                raise ValueError(f'Unable to round: the rank of factors is not lower than the rank of transform!')

    def _check_reduced_rank(self, rank):
        super()._check_reduced_rank(rank)
        self._round_item_features_transform()

    def encode_item_features(self):
        training_items = self.data.index.itemid.training.old.values
        item_features = self.item_features.reindex(training_items, fill_value=[])
        item_one_hot, self.item_features_labels = stack_features(
            item_features, stacked_index=False, normalize=False)
        return item_one_hot

    def update_item_features_transform(self):
        mapping = self.item_features_embeddings
        mapping_invgram = np.linalg.pinv(mapping.T @ mapping)
        self._item_features_transform_helper = mapping_invgram

    def prepare_item_features_transformation(self):
        item_one_hot = self.encode_item_features()
        mapping = self.compute_item_features_mapping(item_one_hot) # model dependent
        item_features_key = f'{self.data.fields.itemid}_features'
        # this will take care of truncating the matrix when the rank is reduced:
        self.factors[item_features_key] = mapping
        self.update_item_features_transform()

    def build(self, *args, **kwargs):
        super().build(*args, return_factors=True, **kwargs)
        self.prepare_item_features_transformation()

    def slice_recommendations(self, cold_item_meta, start, stop):
        cold_slice_meta = cold_item_meta.iloc[start:stop]
        cold_item_features, _ = stack_features(
            cold_slice_meta,
            labels=self.item_features_labels,
            normalize=False)

        u = self.factors[self.data.fields.userid]
        s = self.factors['singular_values']
        w = self.item_features_embeddings
        w_invgram = self._item_features_transform_helper
        cold_items_factors = (cold_item_features @ w) @ w_invgram
        scores = cold_items_factors @ (u * s[None, :]).T
        return scores


class SVDModelItemColdStart(ItemColdStartEvaluationMixin,
                            ItemColdStartRecommenderMixin,
                            ItemColdStartSVDModelMixin,
                            SVDModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'PureSVD(cs)'

    def compute_item_features_mapping(self, item_features):
        itemid = self.data.fields.itemid
        item_factors = self.factors[itemid]
        return item_features.T.dot(item_factors)


class HybridSVDItemColdStart(ItemColdStartEvaluationMixin,
                             ItemColdStartRecommenderMixin,
                             ItemColdStartSVDModelMixin,
                             HybridSVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'HybridSVD(cs)'

    def compute_item_features_mapping(self, item_features):
        itemid = self.data.fields.itemid
        right_projector_key = f'{itemid}_projector_right'
        item_factors = self.factors[right_projector_key]
        return item_features.T.dot(item_factors)


class ScaledSVDItemColdStart(ScaledMatrixMixin, SVDModelItemColdStart): pass


class ScaledHybridSVDItemColdStart(ScaledMatrixMixin, HybridSVDItemColdStart): pass


class ItemColdStartLightFMMixin:
    def slice_recommendations(self, cold_item_meta, start, stop):
        cold_slice_meta = cold_item_meta.iloc[start:stop]
        cold_item_features, _ = stack_features(
            cold_slice_meta,
            labels=self.item_features_labels,
            add_identity=False,
            normalize=True)

        user_embeddings = self._model.user_embeddings
        repr_users = self.data.representative_users
        if repr_users is not None:
            user_embeddings = user_embeddings[repr_users.new.values, :]

        # proper handling of cold-start (instead of built-in predict)
        n_items = self.data.index.itemid.training.shape[0]
        item_features_embeddings = self._model.item_embeddings[n_items:, :]
        cold_items_embeddings = cold_item_features.dot(item_features_embeddings)
        scores = cold_items_embeddings @ user_embeddings.T
        return scores


class LightFMItemColdStart(ItemColdStartEvaluationMixin,
                           ItemColdStartRecommenderMixin,
                           ItemColdStartLightFMMixin,
                           LightFMWrapper): pass
