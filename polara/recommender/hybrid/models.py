import numpy as np

from polara.recommender.models import RecommenderModel
from polara.lib.sparse import sparse_dot


class SimilarityAggregation(RecommenderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'SIM'
        self.implicit = False
        self.dense_output = False
        self.item_similarity_matrix = False

    def build(self):
        # use copy to prevent contaminating original data
        self.item_similarity_matrix = self.data.item_similarity.copy()
        self.item_similarity_matrix.setdiag(0) # exclude self-links
        self.item_similarity_matrix.eliminate_zeros()

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        if self.implicit:
            test_matrix.data = np.ones_like(test_matrix.data)
        scores = sparse_dot(test_matrix, self.item_similarity_matrix, self.dense_output, True)
        return scores, slice_data
