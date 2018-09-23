# python 2/3 interoperability
from __future__ import print_function

import numpy as np
from lightfm import LightFM
from polara.recommender.models import RecommenderModel
from polara.tools.timing import Timer


class LightFMWrapper(RecommenderModel):
    def __init__(self, *args, **kwargs):
        super(LightFMWrapper, self).__init__(*args, **kwargs)
        self.method='LightFM'
        self.rank = 10
        self.fit_method = 'fit_partial'

        self.loss = 'warp'
        self.learning_schedule = 'adagrad'
        self.learning_rate = 0.05
        self.max_sampled = 10

        self.seed = 0
        self._model = None


    def build(self):
        self._model = LightFM(no_components=self.rank,
                              loss=self.loss,
                              learning_rate=self.learning_rate,
                              learning_schedule=self.learning_schedule,
                              max_sampled=self.max_sampled,
                              random_state=self.seed)

        fit = getattr(self._model, self.fit_method)
        matrix = self.get_training_matrix()
        with Timer(self.method, verbose=self.verbose):
            fit(matrix)


    def get_recommendations(self):
        if self.data.warm_start:
            raise NotImplementedError
        else:
            return super(LightFMWrapper, self).get_recommendations()


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        slice_data = self._slice_test_data(test_data, start, stop)
        all_items = data.index.itemid.new.values
        n_users = stop - start
        n_items = len(all_items)
        scores = self._model.predict(np.repeat(test_users[start:stop], n_items),
                                     np.tile(all_items, n_users)).reshape(n_users, n_items)
        return scores, slice_data
