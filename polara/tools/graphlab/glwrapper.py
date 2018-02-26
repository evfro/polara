from polara.recommender.models import RecommenderModel
import graphlab as gl


class GraphlabFactorization(RecommenderModel):
    def __init__(self, *args, **kwargs):
        self.item_side_info = kwargs.pop('item_side_info', None)
        self.user_side_info = kwargs.pop('user_side_info', None)
        super(GraphlabFactorization, self).__init__(*args, **kwargs)
        self._rank = 10
        self.method = 'GLF'
        # side data
        self._item_data = None
        self._user_data = None
        # randomization
        self.seed = 61
        # optimization
        self.binary_target = False
        self.solver = 'auto'
        self.max_iterations = 30
        # reglarization
        self.regularization = 1e-10
        self.linear_regularization = 1e-10
        # sgd
        self.sgd_step_size = 0
        # ranking
        self.ranking_optimization = False
        self.ranking_regularization = 0.25
        self.unobserved_rating_value = None
        self.num_sampled_negative_examples = None
        # other parameters
        self.other_gl_params = {}

    def _on_change(self):
        super(GraphlabFactorization, self)._on_change()
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

                self._item_data = gl.SFrame(side_features)
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

                self._user_data = gl.SFrame(side_features)
        else:
            self._user_data = None
        return self._user_data

    def build(self):
        item_data = self.item_data
        user_data = self.user_data
        side_fact = (item_data is not None) or (user_data is not None)
        params = dict(item_data=item_data,
                      user_data=user_data,
                      side_data_factorization=side_fact,
                      num_factors=self.rank,
                      binary_target=self.binary_target,
                      verbose=self.verbose,
                      # initialization
                      random_seed=self.seed,
                      init_random_sigma=0.1,
                      # optimization
                      solver=self.solver,
                      max_iterations=self.max_iterations,
                      # sgd
                      sgd_step_size=self.sgd_step_size,
                      # regularization
                      regularization=self.regularization,
                      linear_regularization=self.linear_regularization,
                      # other parameters
                      **self.other_gl_params)

        if self.ranking_optimization:
            build_model = gl.ranking_factorization_recommender.create
            params.update(ranking_regularization=self.ranking_regularization,
                          num_sampled_negative_examples=self.num_sampled_negative_examples)
            if self.unobserved_rating_value is not None:
                params.update(unobserved_rating_value=self.unobserved_rating_value)
        else:
            build_model = gl.factorization_recommender.create

        self.gl_model = build_model(gl.SFrame(self.data.training),
                                    user_id=self.data.fields.userid,
                                    item_id=self.data.fields.itemid,
                                    target=self.data.fields.feedback,
                                    **params)
        if self.verbose:
            print '{} training time: {}s'.format(self.method, self.gl_model.training_time)

    def get_recommendations(self):
        userid = self.data.fields.userid
        test_users = self.data.test.evalset[userid].drop_duplicates().values

        recommend = self.gl_model.recommend
        top_recs = recommend(users=test_users,
                             k=self.topk,
                             exclude_known=self.filter_seen,
                             verbose=self.verbose)
        itemid = self.data.fields.itemid
        top_recs = top_recs[itemid].to_numpy().reshape(-1, self.topk)
        return top_recs

    def evaluate_rmse(self):
        feedback = self.data.fields.feedback
        holdout = gl.SFrame(self.data.test.evalset)
        return self.gl_model.evaluate_rmse(holdout, feedback)['rmse_overall']



class WarmStartRecommendationsMixin(object):
    def get_recommendations(self):
        pass


class ColdStartRecommendationsMixin(object):
    def get_recommendations(self):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid

        cold_items_index = self.data.index.itemid.cold_start.old
        lower_index = self.data.index.itemid.training.new.max() + 1
        upper_index = lower_index + len(cold_items_index)
        # prevent intersecting cold items index with known items
        unseen_items_idx = np.arange(lower_index, upper_index)
        new_item_data = gl.SFrame(self.item_side_info.loc[cold_items_index]
                                  .reset_index()
                                  .assign(**{itemid:unseen_items_idx}))

        repr_users = self.data.representative_users.new.values
        observation_idx = [a.flat for a in np.broadcast_arrays(repr_users, unseen_items_idx[:, None])]
        new_observation = gl.SFrame(dict(zip([userid, itemid], observation_idx)))

        scores = self.gl_model.predict(new_observation, new_item_data=new_item_data).to_numpy()
        top_similar_idx = self.get_topk_items(scores.reshape(-1, len(repr_users)))
        top_similar_users = repr_users[top_similar_idx.ravel()].reshape(top_similar_idx.shape)
        return top_similar_users
