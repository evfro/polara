from polara.recommender import defaults
import pandas as pd
import numpy as np
from collections import namedtuple


def random_choice(df, num, random_state):
    return df.iloc[random_state.choice(df.shape[0], num, replace=False)]


def filter_by_length(data, userid='userid', min_session_length=3):
    """Filters users with insufficient number of items"""
    if data.duplicated().any():
        raise NotImplementedError

    sz = data[userid].value_counts(sort=False)
    has_valid_session_length = sz >= min_session_length
    if not has_valid_session_length.all():
        valid_users = sz.index[has_valid_session_length]
        new_data =  data[data[userid].isin(valid_users)].copy()
        print 'Sessions are filtered by length'
    else:
        new_data = data
    return new_data


class RecommenderData(object):
    _std_fields = ('userid', 'itemid', 'feedback')

    _config = {'_shuffle_data', '_test_ratio', '_test_fold',
               '_test_unseen_users', '_holdout_size', '_test_sample',
               '_permute_tops', '_random_holdout', '_negative_prediction'}

    def __init__(self, data, userid, itemid, feedback, custom_order=None):
        self.name = None
        fields_selection = [userid, itemid, feedback]

        if data.duplicated(fields_selection).any():
            #unstable in pandas v. 17.0, only works in <> v.17.0
            #rely on deduplicated data in many places - makes data processing more efficient
            raise NotImplementedError('Data has duplicate values')

        self._custom_order = custom_order
        if self._custom_order:
            fields_selection.append(self._custom_order)

        self._data = data[fields_selection].copy()
        self.fields = namedtuple('Fields', self._std_fields)
        self.fields = self.fields._make(map(eval, self._std_fields))
        self.index = namedtuple('DataIndex', self._std_fields)
        self.index = self.index._make([None]*len(self._std_fields))

        self._set_defaults()
        self._change_properties = set() #container for changed properties
        self.random_state = None #use with shuffle_data, permute_tops, random_choice
        self.verify_sessions_length_distribution = True

        self._attached_models = {'on_change': {}, 'on_update': {}}
        # on_change indicates whether full data has been changed
        # on_update indicates whether only test data has been changed


    def _get_attached_models(self, event):
        return self._attached_models[event]

    def _attach_model(self, event, model, callback):
        self._get_attached_models(event)[model] = callback

    def _detach_model(self, event, model):
        del self._get_attached_models(event)[model]

    def _notify(self, event):
        for model, callback in self._get_attached_models(event).iteritems():
            getattr(model, callback)()


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
        return self._verified_data_property('_holdout_size')

    @holdout_size.setter
    def holdout_size(self, new_value):
        self._update_data_property('_holdout_size', new_value)

    @property
    def random_holdout(self):
        return self._verified_data_property('_random_holdout')

    @random_holdout.setter
    def random_holdout(self, new_value):
        self._update_data_property('_random_holdout', new_value)

    @property
    def permute_tops(self):
        return self._verified_data_property('_permute_tops')

    @permute_tops.setter
    def permute_tops(self, new_value):
        self._update_data_property('_permute_tops', new_value)

    @property
    def negative_prediction(self):
        return self._verified_data_property('_negative_prediction')

    @negative_prediction.setter
    def negative_prediction(self, new_value):
        self._update_data_property('_negative_prediction', new_value)

    @property
    def test_sample(self):
        return self._verified_data_property('_test_sample')

    @test_sample.setter
    def test_sample(self, new_value):
        self._update_data_property('_test_sample', new_value)

    #properties that require rebuilding training and test datasets
    @property
    def shuffle_data(self):
        return self._verified_data_property('_shuffle_data')

    @shuffle_data.setter
    def shuffle_data(self, new_value):
        self._update_data_property('_shuffle_data', new_value)

    @property
    def test_ratio(self):
        return self._verified_data_property('_test_ratio')

    @test_ratio.setter
    def test_ratio(self, new_value):
        self._update_data_property('_test_ratio', new_value)

    @property
    def test_fold(self):
        return self._verified_data_property('_test_fold')

    @test_fold.setter
    def test_fold(self, new_value):
        max_fold = 1.0 / self._test_ratio
        if new_value > max_fold:
            raise ValueError('Test fold value cannot be greater than {}'.format(max_fold))
        self._update_data_property('_test_fold', new_value)

    @property
    def test(self):
        self.update()
        return self._test

    @property
    def training(self):
        self.update() # both _test and _training attributes appear simultaneously
        return self._training


    def _lazy_data_update(self, data_property):
        self._change_properties.add(data_property)


    def _update_data_property(self, data_property, new_value):
        old_value = getattr(self, data_property)
        if old_value != new_value:
            setattr(self, data_property, new_value)
            self._lazy_data_update(data_property)


    def _verified_data_property(self, data_property):
        if data_property in self._change_properties:
            print 'The value of {} might be not effective yet.'.format(data_property[1:])
        return getattr(self, data_property)


    def update(self):
        if self._change_properties:
            self.prepare()


    def prepare(self):
        print 'Preparing data'
        if self._shuffle_data:
            self._data = self._data.sample(frac=1, random_state=self.random_state)
        elif '_shuffle_data' in self._change_properties:
            print 'Recovering original data state due to change in shuffle_data.'
            self._data = self._data.sort_index()

        self._change_properties.clear()

        self._split_test_data()
        self._reindex_data()
        self._align_test_items()
        self._split_eval_data()

        self._notify('on_change')
    def _split_test_index(self):
        # check that folds' sizes will be balanced (in terms of a number of items)
        user_sessions_size, user_idx = self._get_sessions_info()
        n_users = len(user_sessions_size)
        test_split = self._split_test_users(user_idx, n_users, self._test_fold, self._test_ratio)
        return test_split


    def _get_sessions_info(self):
        userid = self.fields.userid
        user_sessions = self._data.groupby(userid, sort=True) #KEEP TRUE HERE!
        # if False than long sessions idx are prevalent in the beginning => non-equal size folds
        # this effect is taken into account with help of is_not_uniform function
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
        if self.verify_sessions_length_distribution:
            if self.is_not_uniform(user_idx):
                print 'Users are not uniformly ordered! Unable to split test set reliably.'
            self.verify_sessions_length_distribution = False
        user_sessions_len = user_sessions.size()
        return user_sessions_len, user_idx


    @staticmethod
    def is_not_uniform(idx, nbins=10, allowed_gap=0.75):
        idx_bins = pd.cut(idx, bins=nbins, labels=False)
        idx_bin_size = np.bincount(idx_bins)

        diff = idx_bin_size[:-1] - idx_bin_size[1:]
        monotonic = (diff < 0).all() or (diff > 0).all()
        huge_gap = (idx_bin_size.min()*1.0 / idx_bin_size.max()) < allowed_gap
        return monotonic or huge_gap


    @staticmethod
    def _split_test_users(idx, n_users, fold, ratio):
        num = n_users * ratio
        selection = (idx >= round((fold-1) * num)) & (idx < round(fold * num))
        return selection


    def _try_reindex_training_data(self):
        if self.build_index:
            self._reindex_train_users()
            self._reindex_train_items()
            self._reindex_feedback()
    def _reindex_train_users(self):
        userid = self.fields.userid
        user_index = self.reindex(self._training, userid, sort=False)
        user_index = namedtuple('UserIndex', 'training test')._make([user_index, None])
        self.index = self.index._replace(userid=user_index)

    def _reindex_train_items(self):
        itemid = self.fields.itemid
        items_index = self.reindex(self._training, itemid)
        self.index = self.index._replace(itemid=items_index)

    def _reindex_feedback(self):
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
        items_index = self.index.itemid.set_index('old')
        itemid = self.fields.itemid
        #this changes int to float dtype if NaN values exist:
        self._test.loc[:, itemid] = items_index.loc[self._test[itemid].values, 'new'].values
        # need to filter those items which were not in the training set
        unseen_items = self._test[itemid].isnull()
        if unseen_items.any():
            userid = self.fields.userid
            test_data = self._test[~unseen_items]
            # there could be insufficient data now - check again
            valid_users_sel = test_data.groupby(userid, sort=False).size() > self._holdout_size
            if not valid_users_sel.all():
                nfiltered = (~valid_users_sel).sum()
                valid_users_sel = valid_users_sel[valid_users_sel]
                print '{} test users filtered due to insufficient number of seen items.'.format(nfiltered)
            else:
                print '{} unseen items filtered from testset.'.format(unseen_items.sum())
            valid_users_idx = valid_users_sel.index[valid_users_sel]

            self._test = test_data.loc[test_data[userid].isin(valid_users_idx)].copy()
            #force int dtype after filtering NaNs
            self._test[itemid] = self._test[itemid].astype(np.int64)
            #reindex the test userids as they were filtered
            new_test_idx = self.reindex(self._test, userid, sort=False, inplace=True)
            #update index info accordingly
            old_test_idx = self.index.userid.test
            self.index = self.index._replace(userid=self.index.userid._replace(test=old_test_idx[old_test_idx['new'].isin(valid_users_idx)]))
            self.index.userid.test.loc[new_test_idx['old'].values, 'new'] = new_test_idx['new'].values


    def _split_eval_data(self):
        userid, feedback = self.fields.userid, self.fields.feedback

        if self._change_properties: #
            print 'Updating test data.'
            self._test = self._test_old

            self._notify('on_update')
            self._change_properties.clear()

        # data may have many items with top ratings and result depends on how
        # they are sorted. randomizing the data helps to avoid biases
        if self.permute_tops:
            test_data = self._test.sample(frac=1, random_state=self.random_state)
        else:
            test_data = self._test

        # split holdout items from the rest of the test data
        holdout = self._sample_holdout(test_data)
        evalidx = holdout.index.get_level_values(1)
        evalset = test_data.loc[evalidx]

        # get test users whos items were hidden
        testset = test_data[~test_data.index.isin(evalidx)]
        # leave at most self.test_sample items for every test user
        testset = self._sample_testset(testset)

        # ensure identical ordering of users in testset and holdout
        testset = testset.sort_values(userid)
        evalset = evalset.sort_values(userid)

        self._test_old = self._test
        #TODO make it computed from index data and _data
        #self._test_idx = namedtuple('TestDataIndex', 'testset evalset')
        #           ._make([self._test.testset.index, self._test.evalset.index])
        self._test = namedtuple('TestData', 'testset evalset')._make([testset, evalset])


    def _sample_holdout(self, data):
        userid, feedback = self.fields.userid, self.fields.feedback
        order_field = self._custom_order or feedback
        grouper = data.groupby(userid, sort=False)[order_field]

        if self.random_holdout: #randomly sample data for evaluation
            holdout = grouper.apply(random_choice, self._holdout_size, self.random_state or np.random)
        elif self.negative_prediction: #try to holdout negative only examples
            holdout = grouper.nsmallest(self._holdout_size)
        else: #standard top-score prediction mode
            holdout = grouper.nlargest(self._holdout_size)

        return holdout


    def _sample_testset(self, data):
        test_sample = self.test_sample
        if not isinstance(test_sample, int):
            return data

        userid, feedback = self.fields.userid, self.fields.feedback
        if test_sample > 0:
            sampled = (data.groupby(userid, sort=False, group_keys=False)
                            .apply(random_choice, test_sample, self.random_state or np.random))
        elif test_sample < 0: #leave only the most negative feedback from user
            idx = (data.groupby(userid, sort=False)[feedback]
                        .nsmallest(-test_sample).index.get_level_values(1))
            sampled = data.loc[idx]
        else:
            sampled = data

        return sampled


    def to_coo(self, tensor_mode=False):
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
        idx = idx.astype(np.intp)
        val = np.ascontiguousarray(val)
        return idx, val, shp


    def test_to_coo(self, tensor_mode=False):
        userid, itemid, feedback = self.fields
        test_data = self.test.testset

        user_idx = test_data[userid].values.astype(np.intp)
        item_idx = test_data[itemid].values.astype(np.intp)
        fdbk_val = test_data[feedback].values

        if tensor_mode:
            fdbk_idx = self.index.feedback.set_index('old').loc[fdbk_val, 'new'].values
            if np.isnan(fdbk_idx).any():
                raise NotImplementedError('Not all values of feedback are present in training data')
            else:
                fdbk_idx = fdbk_idx.astype(np.intp)
            test_coo = (user_idx, item_idx, fdbk_idx)
        else:
            test_coo = (user_idx, item_idx, fdbk_val)

        return test_coo


    def get_test_shape(self, tensor_mode=False):
        #TODO make it a property maybe
        userid = self.fields.userid
        num_users = self.test.testset[userid].max() + 1
        num_items = len(self.index.itemid)
        shape = (num_users, num_items)

        if tensor_mode:
            num_fdbks = len(self.index.feedback)
            shape = shape + (num_fdbks,)

        return shape


class BinaryDataMixin(object):
    def __init__(self, *args, **kwargs):
        self.binary_threshold = kwargs.pop('binary_threshold', None)
        super(BinaryDataMixin, self).__init__(*args, **kwargs)

    def _binarize(self, data, return_filtered_users=False):
        feedback = self.fields.feedback
        data = data[data[feedback] >= self.binary_threshold].copy()
        data[feedback] = np.ones_like(data[feedback])
        return data

    def _split_test_data(self):
        super(BinaryDataMixin, self)._split_test_data()
        if self.binary_threshold is not None:
            self._training = self._binarize(self._training)

    def _split_eval_data(self):
        super(BinaryDataMixin, self)._split_eval_data()
        if self.binary_threshold is not None:
            userid = self.fields.userid
            testset = self._binarize(self.test.testset)
            test_users = testset[userid].unique()
            user_sel = self.test.evalset[userid].isin(test_users)
            evalset = self.test.evalset[user_sel].copy()
            self._test = namedtuple('TestData', 'testset evalset')._make([testset, evalset])
            if len(test_users) != (testset[userid].max()+1):
                # remove gaps in test user indices
                self._update_test_user_index()

    def _update_test_user_index(self):
        testset, evalset = self._test
        userid = self.fields.userid
        new_test_idx = self.reindex(testset, userid, sort=False, inplace=True)
        evalset.loc[:, userid] = evalset[userid].map(new_test_idx.set_index('old').new)
        new_test_idx.old = new_test_idx.old.map(self.index.userid.test.set_index('new').old)
        self.index = self.index._replace(userid=self.index.userid._replace(test=new_test_idx))


class LongTailMixin(object):
    def __init__(self, *args, **kwargs):
        self.long_tail_holdout = kwargs.pop('long_tail_holdout', False)
        # use predefined list if defined
        self.short_head_items = kwargs.pop('short_head_items', None)
        # amount of feedback accumulated in short head
        self.head_feedback_frac = kwargs.pop('head_feedback_frac', 0.33)
        # fraction of popular items considered as short head
        self.head_items_frac = kwargs.pop('head_items_frac', None)
        self._long_tail_items = None
        super(LongTailMixin, self).__init__(*args, **kwargs)

    @property
    def long_tail_items(self):
        if self.short_head_items is not None:
            short_head = self.short_head_items
            long_tail = self.index.itemid.query('old not in @short_head').new.values
        else:
            long_tail = self._get_long_tail()
        return long_tail

    def _get_long_tail(self):
        itemid = self.fields.itemid
        popularity = self.training[itemid].value_counts(ascending=False, normalize=True)
        tail_idx = None

        if self.head_items_frac:
            self.head_feedback_frac = None # could in principle calculate real value instead
            items_frac = np.arange(1, len(popularity)+1) / len(popularity)
            tail_idx = items_frac > self.head_items_frac

        if self.head_feedback_frac:
            tail_idx = popularity.cumsum().values > self.head_feedback_frac

        if tail_idx is None:
            long_tail = None
            self.long_tail_holdout = False
        else:
            long_tail = popularity.index[tail_idx]

        return long_tail

    def _sample_holdout(self, data):
        if self.long_tail_holdout:
            itemid = self.fields.itemid
            long_tail_sel = data[itemid].isin(self.long_tail_items)
            self.__head_data = data[~long_tail_sel]
            data = data[long_tail_sel]
        return super(LongTailMixin, self)._sample_holdout(data)

    def _sample_test_data(self, data):
        if self.long_tail_holdout:
            data = pd.concat([self.__head_data, data], copy=True)
            del self.__head_data
        return super(LongTailMixin, self)._sample_test_data(data)
