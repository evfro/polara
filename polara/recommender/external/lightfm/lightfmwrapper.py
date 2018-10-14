# python 2/3 interoperability
from __future__ import print_function

import numpy as np
from numpy.lib.stride_tricks import as_strided
from lightfm import LightFM
from polara.recommender.models import RecommenderModel
from polara.lib.similarity import stack_features
from polara.tools.timing import Timer


class LightFMWrapper(RecommenderModel):
    def __init__(self, *args, item_features=None, user_features=None, **kwargs):
        super(LightFMWrapper, self).__init__(*args, **kwargs)
        self.method='LightFM'
        self.rank = 10
        self.fit_method = 'fit'

        self.item_features = item_features
        self.item_feature_labels = None
        self.item_alpha = 0.0
        self.item_identity = True
        self._item_features_csr = None

        self.user_features = user_features
        self.user_feature_labels = None
        self.user_alpha = 0.0
        self.user_identity = True
        self._user_features_csr = None

        self.loss = 'warp'
        self.learning_schedule = 'adagrad'
        self.learning_rate = 0.05
        self.max_sampled = 10

        self.seed = 0
        self._model = None


    def build(self):
        self._model = LightFM(no_components=self.rank,
                              item_alpha=self.item_alpha,
                              user_alpha=self.user_alpha,
                              loss=self.loss,
                              learning_rate=self.learning_rate,
                              learning_schedule=self.learning_schedule,
                              max_sampled=self.max_sampled,
                              random_state=self.seed)
        fit = getattr(self._model, self.fit_method)

        matrix = self.get_training_matrix()

        if self.item_features is not None:
            item_features = self.item_features.reindex(self.data.index.itemid.old.values, fill_value=[])
            self._item_features_csr, self.item_feature_labels = stack_features(item_features,
                                                                               add_identity=self.item_identity,
                                                                               normalize=True,
                                                                               dtype='f4')
        if self.user_features is not None:
            user_features = self.user_features.reindex(self.data.index.userid.training.old.values, fill_value=[])
            self._user_features_csr, self.user_feature_labels = stack_features(user_features,
                                                                                add_identity=self.user_identity,
                                                                                normalize=True,
                                                                                dtype='f4')

        with Timer(self.method, verbose=self.verbose):
            fit(matrix, item_features=self._item_features_csr, user_features=self._user_features_csr)


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        if self.data.warm_start:
            raise NotImplementedError

        slice_data = self._slice_test_data(test_data, start, stop)
        all_items = self.data.index.itemid.new.values
        n_users = stop - start
        n_items = len(all_items)
        # use stride tricks to avoid unnecessary copies of repeated indices
        # have to conform with LightFM's dtype to avoid additional copies
        itemsize = np.dtype('i4').itemsize
        useridx = as_strided(test_users[start:stop].astype('i4', copy=False),
                             (n_users, n_items), (itemsize, 0))
        itemidx = as_strided(all_items.astype('i4', copy=False),
                             (n_users, n_items), (0, itemsize))
        scores = self._model.predict(useridx.ravel(), itemidx.ravel(),
                                     user_features=self._user_features_csr,
                                     item_features=self._item_features_csr
                                     ).reshape(n_users, n_items)
        return scores, slice_data
