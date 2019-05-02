import numpy as np
import turicreate as tc
from polara import RecommenderModel


class TuriFactorizationRecommender(RecommenderModel):
    def __init__(self, *args, **kwargs):
        self.item_side_info = kwargs.pop('item_side_info', None)
        self.user_side_info = kwargs.pop('user_side_info', None)
        super().__init__(*args, **kwargs)
        self.tc_model = None
        self._rank = 10
        self.method = 'TCF'
        # side data
        self._item_data = None
        self._user_data = None
        self.side_data_factorization = True
        # randomization
        self.seed = 61
        # optimization
        self.binary_target = False
        self.solver = 'auto'
        self.max_iterations = 25
        # regularization
        self.regularization = 1e-10
        self.linear_regularization = 1e-10
        # adagrad
        self.adagrad_momentum_weighting = 0.9
        # sgd
        self.sgd_step_size = 0
        # ranking
        self.ranking_optimization = False
        self.ranking_regularization = 0.25
        self.unobserved_rating_value = None
        self.num_sampled_negative_examples = 4
        # other parameters
        self.with_data_feedback = True
        self.other_tc_params = {}
        self.data.subscribe(self.data.on_change_event, self._clean_metadata)

    def _clean_metadata(self):
        self._item_data = None
        self._user_data = None

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, new_value):
        if new_value != self._rank:
            self._rank = new_value
            self._is_ready = False
            self._recommendations = None

    num_factors = rank # convenience

    @property
    def item_data(self):
        if self.item_side_info is not None:
            if self._item_data is None:
                itemid = self.data.fields.itemid
                item_side_info = self.item_side_info
                index_data = getattr(self.data.index, 'itemid')

                try:
                    item_index = index_data.training
                except AttributeError:
                    item_index = index_data

                index_map = item_index.set_index('old').new
                side_features = (item_side_info.loc[item_index['old']]
                                               .reset_index())
                side_features[itemid] = side_features[itemid].map(index_map)

                self._item_data = tc.SFrame(side_features)
        else:
            self._item_data = None
        return self._item_data

    @property
    def user_data(self):
        if self.user_side_info is not None:
            if self._user_data is None:
                userid = self.data.fields.userid
                user_side_info = self.user_side_info
                index_data = getattr(self.data.index, 'userid')

                try:
                    user_index = index_data.training
                except AttributeError:
                    user_index = index_data

                index_map = user_index.set_index('old').new
                side_features = (user_side_info.loc[user_index['old']]
                                               .reset_index())
                side_features[userid] = side_features[userid].map(index_map)

                self._user_data = tc.SFrame(side_features)
        else:
            self._user_data = None
        return self._user_data

    def build(self):
        item_data = self.item_data
        user_data = self.user_data
        params = dict(item_data=item_data,
                      user_data=user_data,
                      side_data_factorization=self.side_data_factorization,
                      num_factors=self.rank,
                      binary_target=self.binary_target,
                      verbose=self.verbose,
                      # initialization
                      random_seed=self.seed,
                      init_random_sigma=0.1,
                      # optimization
                      solver=self.solver,
                      max_iterations=self.max_iterations,
                      # adagrad
                      adagrad_momentum_weighting=self.adagrad_momentum_weighting,
                      # sgd
                      sgd_step_size=self.sgd_step_size,
                      # regularization
                      regularization=self.regularization,
                      linear_regularization=self.linear_regularization,
                      # other parameters
                      **self.other_tc_params)

        if self.ranking_optimization:
            build_model = tc.recommender.ranking_factorization_recommender.create
            params.update(ranking_regularization=self.ranking_regularization,
                          num_sampled_negative_examples=self.num_sampled_negative_examples)
            if self.unobserved_rating_value is not None:
                params.update(unobserved_rating_value=self.unobserved_rating_value)
        else:
            build_model = tc.factorization_recommender.create

        target = self.data.fields.feedback if self.with_data_feedback else None
        self.tc_model = build_model(tc.SFrame(self.data.training),
                                    user_id=self.data.fields.userid,
                                    item_id=self.data.fields.itemid,
                                    target=target,
                                    **params)
        if self.training_time is not None:
            self.training_time.append(self.tc_model.training_time)
        if self.verbose:
            print(f'{self.method} training time: {self.tc_model.training_time}s')

    def get_recommendations(self):
        if self.data.warm_start:
            raise NotImplementedError

        userid = self.data.fields.userid
        test_users = self.data.test.holdout[userid].drop_duplicates().values

        top_recs =  self.tc_model.recommend(users=test_users,
                                            k=self.topk,
                                            exclude_known=self.filter_seen,
                                            verbose=self.verbose)
        itemid = self.data.fields.itemid
        top_recs = top_recs[itemid].to_numpy().reshape(-1, self.topk)
        return top_recs

    def evaluate_rmse(self):
        if self.data.warm_start:
            raise NotImplementedError
        feedback = self.data.fields.feedback
        holdout = tc.SFrame(self.data.test.holdout)
        return self.tc_model.evaluate_rmse(holdout, feedback)['rmse_overall']



class WarmStartRecommendationsMixin:
    def get_recommendations(self):
        pass


class ColdStartRecommendationsMixin:
    def get_recommendations(self):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        data_index = self.data.index

        cold_items_index = data_index.itemid.cold_start.old.values
        lower_index = data_index.itemid.training.new.max() + 1
        upper_index = lower_index + len(cold_items_index)
        # prevent intersecting cold items index with known items
        unseen_items_idx = np.arange(lower_index, upper_index)
        new_item_data = tc.SFrame(self.item_side_info.loc[cold_items_index]
                                  .reset_index()
                                  .assign(**{itemid: unseen_items_idx}))
        repr_users = self.data.representative_users
        try:
            repr_users = repr_users.new.values
        except AttributeError:
            repr_users = data_index.userid.training.new.values
        observation_idx = [a.flat for a in np.broadcast_arrays(repr_users, unseen_items_idx[:, None])]
        new_observation = tc.SFrame(dict(zip([userid, itemid], observation_idx)))

        scores = self.tc_model.predict(new_observation, new_item_data=new_item_data).to_numpy()
        top_similar_idx = self.get_topk_elements(scores.reshape(-1, len(repr_users)))
        top_similar_users = repr_users[top_similar_idx.ravel()].reshape(top_similar_idx.shape)
        return top_similar_users
