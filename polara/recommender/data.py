from polara.recommender import defaults
import pandas as pd
import numpy as np
from collections import namedtuple


class RecommenderData(object):
    _std_fields = ('userid', 'itemid', 'feedback')
    _datawise_properties = {'_shuffle_data', '_test_ratio', '_test_fold'}
    _testwise_properties = {'_holdout_size', '_test_sample', '_permute_tops', '_random_holdout', '_negative_prediction'}
    _config = _datawise_properties.union(_testwise_properties)
    # ('test_ratio', 'holdout_size', 'test_fold', 'shuffle_data',
    #             'test_sample', 'permute_tops', 'random_holdout', 'negative_prediction')

    def __init__(self, data, userid, itemid, feedback):
        self.name = None
        if data.duplicated([userid, itemid]).any():
            #unstable in pandas v. 17.0, only works in <> v.17.0
            #rely on deduplicated data in many places - makes data processing more efficient
            raise NotImplementedError('Data has duplicate values')

        self._data = data[[userid, itemid, feedback]].copy()
        self.fields = namedtuple('Fields', self._std_fields)
        self.fields = self.fields._make(map(eval, self._std_fields))
        self.index = namedtuple('DataIndex', self._std_fields)
        self.index = self.index._make([None]*len(self._std_fields))

        self._set_defaults()
        self._has_updated = False #indicated whether test data has been changed
        self._has_changed = False #indicated whether full data has been changed
        self._change_properties = set() #container for changed properties
        self.random_seed = None #use with shuffle_data and permute_tops property


    def _set_defaults(self, params=None):
        #[1:] omits undersacores in properties names
        params = params or [prop[1:] for prop in self._config]
        config_vals = defaults.get_config(params)
        for name, value in config_vals.iteritems():
            internal_name = '_{}'.format(name)
            setattr(self, internal_name, value)


    def get_configuration(self):
        #[1:] omits undersacores in properties names, i.e. uses external name
        #in that case it prints worning if change is pending
        config = {attr[1:]: getattr(self, attr[1:]) for attr in self._config}
        return config


    #properties that change evaluation set but do not require rebuilding test data
    @property
    def holdout_size(self):
        # return self._holdout_size
        return self._verified_data_property('_holdout_size')

    @holdout_size.setter
    def holdout_size(self, new_value):
        self._update_data_property('_holdout_size', new_value)

    @property
    def random_holdout(self):
        # return self._random_holdout
        return self._verified_data_property('_random_holdout')

    @random_holdout.setter
    def random_holdout(self, new_value):
        self._update_data_property('_random_holdout', new_value)

    @property
    def permute_tops(self):
        # return self._permute_tops
        return self._verified_data_property('_permute_tops')

    @permute_tops.setter
    def permute_tops(self, new_value):
        self._update_data_property('_permute_tops', new_value)

    @property
    def negative_prediction(self):
        # return self._negative_prediction
        return self._verified_data_property('_negative_prediction')

    @negative_prediction.setter
    def negative_prediction(self, new_value):
        self._update_data_property('_negative_prediction', new_value)

    @property
    def test_sample(self):
        # return self._test_sample
        return self._verified_data_property('_test_sample')

    @test_sample.setter
    def test_sample(self, new_value):
        self._update_data_property('_test_sample', new_value)

    #properties that require rebuilding training and test datasets
    @property
    def shuffle_data(self):
        # return self._shuffle_data
        return self._verified_data_property('_shuffle_data')

    @shuffle_data.setter
    def shuffle_data(self, new_value):
        self._update_data_property('_shuffle_data', new_value)

    @property
    def test_ratio(self):
        # return self._test_ratio
        return self._verified_data_property('_test_ratio')

    @test_ratio.setter
    def test_ratio(self, new_value):
        self._update_data_property('_test_ratio', new_value)

    @property
    def test_fold(self):
        # return self._test_fold
        return self._verified_data_property('_test_fold')

    @test_fold.setter
    def test_fold(self, new_value):
        max_fold = 1.0 / self._test_ratio
        if new_value > max_fold:
            raise ValueError('Test fold value cannot be greater than {}'.format(max_fold))
        self._update_data_property('_test_fold', new_value)

    @property
    def test(self):
        update_data, update_test = self._pending_change()
        if update_data or not hasattr(self, '_test'):
            self.prepare()
            test_change_required = False #changes are already effective
        if update_test:
            self._split_eval_data()
        return self._test

    @property
    def training(self):
        update_data, _ = self._pending_change()
        if update_data or not hasattr(self, '_training'):
            self.prepare()
        return self._training

    @property
    def has_changed(self):
        value = self._has_changed
        self._has_changed = False #this is an indicator property, reset once read
        return value

    @property
    def has_updated(self):
        value = self._has_updated
        self._has_updated = False #this is an indicator property, reset once read
        return value


    def _lazy_data_update(self, data_property):
        self._change_properties.add(data_property)


    def _update_data_property(self, data_property, new_value):
        old_value = getattr(self, data_property)
        if old_value != new_value:
            setattr(self, data_property, new_value)
            self._lazy_data_update(data_property)


    def _verified_data_property(self, data_property):
        #update_data, update_test = self._pending_change(data_property)
        if data_property in self._change_properties:
            print 'The value of {} might be not effective yet.'.format(data_property[1:])
        return getattr(self, data_property)


    def _pending_change(self, data_properties=None):
        update_data = self._change_properties - self._testwise_properties
        update_test = self._change_properties - self._datawise_properties

        if data_properties:
            data_properties = set(data_properties)
            update_data = update_data.intersection_update(data_properties)
            update_test = update_test.intersection_update(data_properties)

        return update_data, update_test


    def prepare(self):
        print 'Preparing data'
        if self._shuffle_data:
            self._data = self._data.sample(frac=1, random_state=self.random_seed)
        elif '_shuffle_data' in self._change_properties:
            print 'Recovering original data state due to change in shuffle_data.'
            self._data = self._data.sort_index()
        self._change_properties.clear()

        self._split_test_data()
        self._reindex_data()
        self._align_test_items()
        self._split_eval_data()

        self._has_changed = True
        #TODO implement operations with this container


    def update(self):
        data_update_pending, test_update_pending = self._pending_change()
        if data_update_pending or not hasattr(self, '_test'):
            self.prepare()
            test_update_pending = False
        if test_update_pending:
            self._split_eval_data()


    @staticmethod
    def is_not_uniform(idx, nbins=10, allowed_gap=0.75):
        idx_bins = pd.cut(idx, bins=nbins, labels=False)
        idx_bin_size = np.bincount(idx_bins)

        diff = idx_bin_size[:-1] - idx_bin_size[1:]
        monotonic = (diff < 0).all() or (diff > 0).all()
        huge_gap = (idx_bin_size.min()*1.0 / idx_bin_size.max()) < allowed_gap
        return monotonic or huge_gap


    def _split_test_data(self):
        userid, itemid = self.fields.userid, self.fields.itemid
        test_fold = self._test_fold

        user_sessions = self._data.groupby(userid, sort=True) #KEEP TRUE HERE!!!!
        # if False than long sessions idx are prevalent in the beginning => non-equal size folds
        # this is accounted for with help of is_not_uniform function
        # example (run several times to see a pattern):
        # df = pd.DataFrame(index=range(10),
        #                    data={'A':list('abbcceefgg'),
        #                          'N':[1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        #                          'O':range(10)})
        # sampled = df.sample(frac=1)
        # print_frames((df.T, sampled.T))
        # idx_false = df.groupby('N', sort=False).grouper.group_info[0]
        # print idx_false
        # idx_sample_false = sampled.groupby('N', sort=False).grouper.group_info[0]
        # print idx_sample_false
        # idx_orig = df.groupby('N', sort=True).grouper.group_info[0]
        # print_frames((sampled[idx_sample_false>1],
        #               sampled[idx_sample_true>1], df[idx_orig>1]))

        user_idx = user_sessions.grouper.group_info[0]
        is_skewed = self.is_not_uniform(user_idx)
        if is_skewed:
            print 'Users are not uniformly ordered! Unable to split test set reliably.'
            #raise NotImplementedError('Users are not uniformly ordered! Unable to split test set reliably.')

        user_sessions_len = user_sessions.size()
        if (user_sessions_len <= self._holdout_size).any():
            raise NotImplementedError('Some users have not enough items for evaluation')

        user_num = user_sessions_len.size #number of unique users
        test_user_num = user_num * self._test_ratio

        left_condition = user_idx < round((test_fold-1) * test_user_num)
        right_condition = user_idx >= round(test_fold * test_user_num)
        training_selection = left_condition | right_condition
        test_selection = ~training_selection

        self._training = self._data[training_selection].copy()
        self._test = self._data[test_selection].copy()


    def _reindex_data(self):
        userid, itemid, feedback = self.fields
        reindex = self.reindex
        user_index = [reindex(data, userid, sort=False) for data in [self._training, self._test]]
        user_index = namedtuple('UserIndex', 'training test')._make(user_index)
        self.index = self.index._replace(userid=user_index)
        self.index = self.index._replace(itemid=reindex(self._training, itemid))
        self.index = self.index._replace(feedback=None)


    @staticmethod
    def reindex(data, col, sort=True, inplace=True):
        grouper = data.groupby(col, sort=sort).grouper
        new_val = grouper.group_info[1]
        old_val = grouper.levels[0]
        val_transform = pd.DataFrame({'old': old_val, 'new': new_val})
        new_data = grouper.group_info[0]

        if inplace:
            result = val_transform
            data.loc[:, col] = new_data
        else:
            result = (new_data, val_transform)
        return result


    def _align_test_items(self):
        #TODO: add option to filter by whole sessions, not just items
        items_index = self.index.itemid.set_index('old')
        itemid = self.fields.itemid

        self._test.loc[:, itemid] = items_index.loc[self._test[itemid].values, 'new'].values
        # need to filter those items which were not in the training set
        unseen_items_num = self._test[itemid].isnull().sum()

        if unseen_items_num > 0:
            #'%i unseen items found in the test set. Dropping...' % unseen_items_num
            userid = self.fields.userid
            self._test.dropna(axis=0, subset=[itemid], inplace=True)

            # there could be insufficient data now - check again
            valid_users_sel = self._test.groupby(userid, sort=False).size() > self._holdout_size
            if (~valid_users_sel).any():
                raise NotImplementedError('Some users have not enough items for evaluation')
            valid_users_idx = valid_users_sel.index[valid_users_sel]

            self._test = self._test.loc[self._test[userid].isin(valid_users_idx)]
            self._test[itemid] = self._test[itemid].astype(np.int64)
            #reindex the test userids as they were filtered
            new_test_idx = self.reindex(self._test, userid, sort=False, inplace=True)
            #update index info accordingly
            old_test_idx = self.index.userid.test
            self.index.userid._replace(test=old_test_idx[old_test_idx['new'].isin(valid_users_idx)])
            self.index.userid.test.loc[new_test_idx['old'].values, 'new'] = new_test_idx['new'].values


    def _split_eval_data(self, renew=False):
        def random_choice(df, num):
            # TODO add control with RandomState
            return df.iloc[np.random.choice(df.shape[0], num, replace=False)]

        if self._change_properties: #
            print 'Updating test data.'
            #print 'preparing new test and eval data'
            self._test = self._test_old
            self._has_updated = True
            self._change_properties.clear()

        userid, itemid, feedback = self.fields
        lastn = self._holdout_size
        test_sample = self._test_sample

        # data may have many items with top ratings and result depends on how
        # they are sorted. randomizing the data helps to avoid biases
        if self.permute_tops:
            test_data = self._test.sample(frac=1, random_state=self.random_seed)
        else:
            test_data = self._test

        eval_grouper = test_data.groupby(userid, sort=False)[feedback]

        if self.random_holdout: #randomly sample data for evaluation
            eval_idx = eval_grouper.apply(random_choice, lastn).index.get_level_values(1)
        elif self.negative_prediction: #try to holdout negative only examples
            eval_idx = eval_grouper.nsmallest(lastn).index.get_level_values(1)
        else: #standard top-score prediction mode
            eval_idx = eval_grouper.nlargest(lastn).index.get_level_values(1)

        #ensure correct sorting of users in test and eval - order must be the same
        evalset = test_data.loc[eval_idx].sort_values(userid)
        testset = test_data[~test_data.index.isin(eval_idx)].sort_values(userid)

        if isinstance(test_sample, int):
            if test_sample > 0:
                testset = (testset.groupby(userid, sort=False, group_keys=False)
                                    .apply(random_choice, test_sample))
            elif test_sample < 0: #leave only the most negative feedback from user
                test_idx = (testset.groupby(userid, sort=False)[feedback]
                                    .nsmallest(-test_sample).index.get_level_values(1))
                testset = testset.loc[test_idx]

        self._test_old = self._test
        #TODO make it computed from index data and _data, instead of storing in memory
        self._test = namedtuple('TestData', 'testset evalset')._make([testset, evalset])
        #self._test_idx = namedtuple('TestDataIndex', 'testset evalset')._make([self._test.testset.index, self._test.evalset.index])


    def to_coo(self, tensor_mode=True):
        userid, itemid, feedback = self.fields
        user_item_data = self.training[[userid, itemid]].values

        if tensor_mode:
            # TODO this recomputes feedback data every new functon call,
            # but if data has not changed - no need for this, make a property
            new_feedback, feedback_transform = self.reindex(self.training, feedback, inplace=False)
            self.index = self.index._replace(feedback=feedback_transform)

            idx = np.hstack((user_item_data, new_feedback[:, np.newaxis]))
            idx = np.ascontiguousarray(idx)
            val = np.ones(self.training.shape[0],)
        else:
            idx = user_item_data
            val = self.training[feedback].values

        shp = tuple(idx.max(axis=0) + 1)
        idx = idx.astype(np.int64)
        val = np.ascontiguousarray(val)
        return idx, val, shp


class RecommenderDataPositive(RecommenderData):
    def __init__(self, pos_value, *args, **kwargs):
        super(RecommenderDataPositive, self).__init__(*args, **kwargs)
        self.pos_value = pos_value

    def _split_test_data(self):
        super(RecommenderDataPositive, self)._split_test_data()
        self._get_positive_only()

    def _get_positive_only(self):
        userid, feedback = self.fields.userid, self.fields.feedback
        pos_only_data = self._training.loc[self._training[feedback] >= self.pos_value]
        valid_users = pos_only_data.groupby(userid, sort=False).size() > self.holdout_size
        user_idx = valid_users.index[valid_users]
        self._training = pos_only_data[pos_only_data.userid.isin(user_idx)].copy()
