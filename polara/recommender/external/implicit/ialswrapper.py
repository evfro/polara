# python 2/3 interoperability
try:
    range = xrange
except NameError:
    pass

import implicit
import numpy as np

from polara.recommender.models import RecommenderModel
from polara.tools.timing import Timer


class ImplicitALS(RecommenderModel):

    def __init__(self, *args, **kwargs):
        super(ImplicitALS, self).__init__(*args, **kwargs)
        self._rank = 10
        self.alpha = 1
        self.epsilon = 1
        self.weight_func = np.log2
        self.regularization = 0.01
        self.num_threads = 0
        self.num_epochs = 15
        self.method = 'iALS'
        self._model = None

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, new_value):
        if new_value != self._rank:
            self._rank = new_value
            self._is_ready = False
            self._recommendations = None

    @staticmethod
    def confidence(values, alpha=1, weight=None, epsilon=1, dtype='double'):
        '''implementation of a generic confidence-based function'''
        values = weight(values / epsilon) if weight is not None else values / epsilon
        return (alpha * values).astype(dtype)

    def build(self):
        # define iALS model instance
        self._model = implicit.als.AlternatingLeastSquares(factors=self.rank,
                                                           regularization=self.regularization,
                                                           iterations=self.num_epochs,
                                                           num_threads=self.num_threads)

        # prepare input matrix for learning the model
        matrix = self.get_training_matrix()  # user_by_item sparse matrix
        matrix.data = self.confidence(matrix.data, alpha=self.alpha,
                                      weight=self.weight_func, epsilon=self.epsilon)

        with Timer(self.method, verbose=self.verbose):
            # build the model
            # implicit takes item_by_user matrix as input, need to transpose
            self._model.fit(matrix.T)

    def get_recommendations(self):
        recalculate = self.data.warm_start  # used to force folding-in computation
        if recalculate:
            if self.filter_seen is False:
                raise ValueError('The model always filters seen items from results.')
            # prepare test matrix with preferences of unseen users
            matrix, _ = self.get_test_matrix()
            matrix.data = self.confidence(matrix.data, alpha=self.alpha,
                                          weight=self.weight_func, epsilon=self.epsilon)
            num_users = matrix.shape[0]
            users_idx = range(num_users)

            top_recs = np.empty((num_users, self.topk), dtype=np.intp)
            for i, user_row in enumerate(users_idx):
                recs = self._model.recommend(user_row, matrix, N=self.topk, recalculate_user=recalculate)
                top_recs[i, :] = [item for item, _ in recs]
        else:
            top_recs = super(ImplicitALS, self).get_recommendations()
        return top_recs

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        slice_data = self._slice_test_data(test_data, start, stop)

        user_factors = self._model.user_factors[test_users[start:stop], :]
        item_factors = self._model.item_factors
        scores = user_factors.dot(item_factors.T)
        return scores, slice_data
