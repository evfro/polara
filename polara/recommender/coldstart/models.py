import numpy as np
from polara.recommender.models import RecommenderModel
from polara.lib.similarity import stack_features
from polara.lib.sparse import sparse_dot


class ItemColdStartEvaluationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_seen = False # there are no seen entities in cold start
        self._prediction_key = '{}_cold'.format(self.data.fields.itemid)
        self._prediction_target = self.data.fields.userid


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


        top_similar_users = self.get_topk_elements(scores).astype(np.intp)
        return top_similar_users
