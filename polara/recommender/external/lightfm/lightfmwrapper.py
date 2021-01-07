import numpy as np
from numpy.lib.stride_tricks import as_strided
from lightfm import LightFM
from polara.recommender.models import RecommenderModel
from polara.lib.similarity import stack_features
from polara.tools.timing import track_time


class LightFMWrapper(RecommenderModel):
    def __init__(self, *args, item_features=None, user_features=None, **kwargs):
        super(LightFMWrapper, self).__init__(*args, **kwargs)
        self.method='LightFM'
        self._rank = 10
        self.fit_method = 'fit'
        self.fit_params = {}

        self.item_features = item_features
        self.item_features_labels = None
        self.item_alpha = 0.0
        self.item_identity = True
        self._item_features_csr = None
        self.normalize_item_features = True

        self.user_features = user_features
        self.user_features_labels = None
        self.user_alpha = 0.0
        self.user_identity = True
        self._user_features_csr = None
        self.normalize_user_features = True

        self.loss = 'warp'
        self.learning_schedule = 'adagrad'
        self.learning_rate = 0.05
        self.max_sampled = 10

        self.seed = 0
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

        matrix = self.get_training_matrix(sparse_format='coo') # as reqired by LightFM

        try:
            item_index = self.data.index.itemid.training
        except AttributeError:
            item_index = self.data.index.itemid

        if self.item_features is not None:
            item_features = self.item_features.reindex(
                item_index.old.values,
                fill_value=[])
            self._item_features_csr, self.item_features_labels = stack_features(
                item_features,
                add_identity=self.item_identity,
                normalize=self.normalize_item_features,
                dtype='f4')
        if self.user_features is not None:
            user_features = self.user_features.reindex(
                self.data.index.userid.training.old.values,
                fill_value=[])
            self._user_features_csr, self.user_features_labels = stack_features(
                user_features,
                add_identity=self.user_identity,
                normalize=self.normalize_user_features,
                dtype='f4')

        with track_time(self.training_time, verbose=self.verbose, model=self.method):
            fit(matrix, item_features=self._item_features_csr, user_features=self._user_features_csr, **self.fit_params)


    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        if self.data.warm_start:
            raise NotImplementedError('Not supported by LightFM.')

        slice_data = self._slice_test_data(test_data, start, stop)
        all_items = self.data.index.itemid.new.values
        n_users = stop - start
        n_items = len(all_items)
        test_shape = (n_users, n_items)
        test_users_index = test_users[start:stop].astype('i4', copy=False)
        test_items_index = all_items.astype('i4', copy=False)
        # use stride tricks to avoid unnecessary copies of repeated indices
        # have to conform with LightFM's dtype to avoid additional copies
        itemsize = np.dtype('i4').itemsize
        scores = self._model.predict(
            as_strided(test_users_index, test_shape, (itemsize, 0)).ravel(),
            as_strided(test_items_index, test_shape, (0, itemsize)).ravel(),
            user_features=self._user_features_csr,
            item_features=self._item_features_csr,
            num_threads=self.fit_params.get('num_threads', 1)
        ).reshape(test_shape)
        return scores, slice_data
