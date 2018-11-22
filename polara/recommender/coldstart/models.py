import numpy as np

from polara.recommender.models import RecommenderModel


class ContentBasedColdStart(RecommenderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'CB'
        self._prediction_key = '{}_cold'.format(self.data.fields.itemid)
        self._prediction_target = self.data.fields.userid

    def build(self):
        pass

    def get_recommendations(self):
        item_similarity_scores = self.data.cold_items_similarity

        user_item_matrix = self.get_training_matrix()
        user_item_matrix.data = np.ones_like(user_item_matrix.data)

        scores = item_similarity_scores.dot(user_item_matrix.T).tocsr()
        top_similar_users = self.get_topk_elements(scores).astype(np.intp)
        return top_similar_users
