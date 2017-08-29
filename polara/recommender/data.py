from polara.recommender import defaults
import pandas as pd
import numpy as np
from collections import namedtuple
from collections import defaultdict


def random_choice(df, num, random_state):
    n = df.shape[0]
    k = min(num, n)
    return df.iloc[random_state.choice(n, k, replace=False)]

def random_sample(df, frac, random_state):
    return df.sample(frac=frac, random_state=random_state)


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


def property_factory(cls):
    # set class properties in the loop, see
    # https://stackoverflow.com/questions/25371906/python-scope-issue-with-anonymous-lambda-in-metaclass
    # https://stackoverflow.com/questions/27629944/python-metaclass-adding-properties
    def getter(x):
        def wrapped(self):
            return self._verified_data_property(x)
        return wrapped

    def setter(x):
        def wrapped(self, v):
            self._update_data_property(x, v)
        return wrapped

    for p in cls._config:
        setattr(cls, p[1:], property(getter(p), setter(p)))
    return cls


@property_factory
class RecommenderData(object):
    _std_fields = ('userid', 'itemid', 'feedback')

    _config = {'_shuffle_data', '_test_ratio', '_test_fold',
               '_test_unseen_users', '_holdout_size', '_test_sample',
               '_permute_tops', '_random_holdout', '_negative_prediction'}

    def __init__(self, data, userid, itemid, feedback, custom_order=None, seed=None):
        self.name = None
        fields = [userid, itemid, feedback]

        if data is None:
            cols = fields + [custom_order] if custom_order else fields
            self._data = data = pd.DataFrame(columns=cols)
        else:
            self._data = data

        if data.duplicated(subset=fields).any():
            #unstable in pandas v. 17.0, only works in <> v.17.0
            #rely on deduplicated data in many places - makes data processing more efficient
            raise NotImplementedError('Data has duplicate values')

        self._custom_order = custom_order
        self.fields = namedtuple('Fields', self._std_fields)
        self.fields = self.fields._make(map(eval, self._std_fields))
        self.index = namedtuple('DataIndex', self._std_fields)
        self.index = self.index._make([None]*len(self._std_fields))

        self._set_defaults()
        self._change_properties = set() #container for changed properties
        # depending on config. For ex., shuffle_data - full_update,
        # TODO seed may also lead to either full_update or test_update
        # random_holdout - test_update. Need to implement checks
        self.seed = seed #use with permute_tops, random_choice
        self.verify_sessions_length_distribution = True
        self.ensure_consistency = True # drop test entities if not present in training
        self.build_index = True # reindex data, avoid gaps in user and item index
        self._test_selector = None
        self._state = None # None or 1 of {'_': 1, 'H': 11, '|': 2, 'd': 3, 'T': 4}

        self._attached_models = {'on_change': {}, 'on_update': {}}
        self.on_change_event = 'on_change'
        self.on_update_event = 'on_update'
        # on_change indicates whether full data has been changed -> rebuild model
        # on_update indicates whether only test data has been changed -> renew recommendations
        self.verbose = True


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
        if self.verbose:
            print 'Preparing data...'

        update_rule = self._split_data()

        if update_rule['full_update']:
            self._try_reindex_training_data()

        if update_rule['full_update'] or update_rule['test_update']:
            self._try_drop_unseen_test_items() # unseen = not present in training data
            self._try_drop_unseen_test_users() # unseen = not present in training data
            self._try_drop_invalid_test_users() # with too few items and/or if inconsistent between testset and holdout
            self._try_reindex_test_data() # either assign known index, or (if testing for unseen users) reindex
            self._try_sort_test_data()


    def _validate_config(self):
        if self._test_unseen_users and not (self._holdout_size and self._test_ratio):
            raise ValueError('Both holdout_size and test_ratio must be positive when test_unseen_users is set to True')

        assert self._test_ratio < 1, 'Value of test_ratio can\'t be greater than or equal to 1'

        if self._test_ratio:
            max_fold = 1.0 / self._test_ratio
            if self._test_fold  > max_fold:
                raise ValueError('Test fold value cannot be greater than {}'.format(max_fold))


    def _check_state_transition(self):
        test_ratio_change = '_test_ratio' in self._change_properties
        test_fold_change = '_test_fold' in self._change_properties
        test_sample_change = '_test_sample' in self._change_properties
        test_data_change = test_fold_change or test_ratio_change
        holdout_sz_change = '_holdout_size' in self._change_properties
        unseen_usr_change = '_test_unseen_users' in self._change_properties
        permute_change = '_permute_tops' in self._change_properties
        negative_change = ('_negative_prediction' in self._change_properties) and not self._random_holdout
        rnd_holdout_change = '_random_holdout' in self._change_properties
        any_holdout_change = holdout_sz_change or rnd_holdout_change or negative_change or permute_change
        empty_holdout = self._holdout_size == 0
        empty_testset = self._test_ratio == 0
        test_unseen = self._test_unseen_users
        last_state = self._state
        update_rule = defaultdict(bool)
        new_state = last_state

        if unseen_usr_change: # unseen_test_users is reserved for state 4 only!
            if test_unseen:
                new_state = 4
                if (last_state == 11) and not test_data_change:
                    update_rule['test_update'] = True
                else:
                    update_rule['full_update'] = True
            else:
                if empty_holdout:
                    if empty_testset:
                        new_state = 1
                        update_rule['full_update'] = True
                    else:
                        new_state = 11
                        update_key = 'full_update' if test_data_change else 'test_update'
                        update_rule[update_key] = True
                else:
                    update_rule['full_update'] = True
                    if empty_testset:
                        new_state = 2
                    else:
                        new_state = 3
        else: # this assumes that test_unseen_users is consistent with current state!
            if last_state == 1: # hsz = 0, trt = 0, usn = False
                if holdout_sz_change: # hsz > 0
                    new_state = 3 if test_ratio_change else 2
                    update_rule['full_update'] = True
                elif test_ratio_change: # hsz = 0,  trt > 0
                    new_state = 11
                    update_rule['full_update'] = True

            elif last_state == 11: # hsz = 0, trt > 0, usn = False
                if holdout_sz_change: # hsz > 0
                    new_state = 2 if empty_testset else 3
                    update_rule['full_update'] = True
                elif test_data_change: # hsz = 0
                    if empty_testset: # hsz = 0, trt = 0
                        new_state = 1
                    update_rule['full_update'] = True

            elif last_state == 2: # hsz > 0, trt = 0, usn = False
                if test_ratio_change: # trt > 0
                    new_state = 11 if empty_holdout else 3
                    update_rule['full_update'] = True

                elif any_holdout_change: # trt = 0
                    if empty_holdout: # hsz = 0
                        new_state = 1
                    update_rule['full_update'] = True

            elif last_state == 3: # hsz > 0, trt > 0, usn = False
                if test_data_change or any_holdout_change:
                    if empty_holdout:
                        new_state = 1 if empty_testset else 11
                    elif empty_testset: # hsz > 0, trt = 0
                        new_state = 2
                    update_rule['full_update'] = True

            elif last_state == 4: # hsz > 0, trt > 0, usn = True
                if any_holdout_change:
                    if empty_holdout:
                        if test_data_change:
                            new_state = 1 if empty_testset else 11
                            update_rule['full_update'] = True
                        else: # hsz = 0, trt > 0
                            new_state = 11
                            update_rule['test_update'] = True
                    else: # hsz > 0
                        if test_data_change:
                            if empty_testset: # hsz > 0, trt = 0
                                new_state = 2
                            update_rule['full_update'] = True
                        else: # including test_sample_change
                            update_rule['test_update'] = True
                else: # hsz > 0
                    if test_data_change:
                        if empty_testset: # hsz > 0, trt = 0
                            new_state = 2
                        update_rule['full_update'] = True
                    elif test_sample_change:
                        update_rule['test_update'] = True

            else: # initial state
                if empty_holdout:
                    new_state = 1 if empty_testset else 11
                else:
                    if empty_testset: # hsz > 0, trt = 0
                        new_state = 2
                    else: # hsz > 0, trt > 0
                        new_state = 4 if test_unseen else 3
                update_rule['full_update'] = True

        return new_state, update_rule


    def _split_data(self):
        self._validate_config()
        new_state, update_rule = self._check_state_transition()

        full_update = update_rule['full_update']
        test_update = update_rule['test_update']

        if not (full_update or test_update):
            print 'Data is ready. No action was taken.'
            return update_rule

        if self._test_ratio:
            if full_update:
                test_split = self._split_test_index()
            else: #test_update
                test_split = self._test_split
            if self._holdout_size == 0:  # state 11
                testset = holdout = None
                train_split = ~test_split
            else: # state 3 or state 4
                holdout = self._sample_holdout(test_split)

                if self._test_unseen_users: # state 4
                    testset = self._sample_testset(test_split, holdout.index)
                    train_split = ~test_split
                else: # state 3
                    testset = None # will be computed if test data is requested
                    train_split = ~self._data.index.isin(holdout.index)
        else: # test_ratio == 0
            testset = None # will be computed if test data is requested
            test_split = slice(None)

            if self._holdout_size >= 1: # state 2, sample holdout data per each user
                holdout = self._sample_holdout(test_split)
            elif self._holdout_size > 0: # state 2, special case - sample whole data at once
                random_state = np.random.RandomState(self.seed)
                holdout = self._data.sample(frac=self._holdout_size, random_state=random_state)
            else: # state 1
                holdout = None

            train_split = slice(None) if holdout is None else ~self._data.index.isin(holdout.index)

        self._state = new_state
        self._test_split = test_split
        self._test = namedtuple('TestData', 'testset evalset')._make([testset, holdout])

        if full_update:
            self._training = self._data.loc[train_split, list(self.fields)]
            self._notify(self.on_change_event)
        elif test_update:
            self._notify(self.on_update_event)

        self._change_properties.clear()
        return update_rule


    def _split_test_index(self):
        sessions_size, sess_idx = self._get_sessions_info()
        n_sessions = len(sessions_size)
        test_split = self._split_fold_index(sess_idx, n_sessions, self._test_fold, self._test_ratio)
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
        # check that folds' sizes will be balanced (in terms of a number of items)
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
    def _split_fold_index(idx, n_unique, fold, ratio):
        # supports both [0, 1, 2, 3] and [0, 0, 1, 1, 1, 2, 3, 3] types of idx
        # if idx contains only unique elements (1 case) then n_unique = len(idx)
        num = n_unique * ratio
        selection = (idx >= round((fold-1) * num)) & (idx < round(fold * num))
        return selection


    def _try_reindex_training_data(self):
        if self.build_index:
            self._reindex_train_users()
            self._reindex_train_items()
            self._reindex_feedback()

    def _try_drop_unseen_test_items(self):
        if self.ensure_consistency:
            itemid = self.fields.itemid
            self._filter_unseen_entity(itemid, self._test.testset, 'testset')
            self._filter_unseen_entity(itemid, self._test.evalset, 'holdout')

    def _try_drop_unseen_test_users(self):
        if self.ensure_consistency: # even in state 3 there could be unseen users
            userid = self.fields.userid
            self._filter_unseen_entity(userid, self._test.evalset, 'holdout')

    def _try_drop_invalid_test_users(self):
        if self.holdout_size >= 1:
            self._filter_short_sessions() # ensure holdout conforms the holdout_size attribute
        self._align_test_users() # ensure the same users are in both testset and holdout

    def _try_reindex_test_data(self):
        self._assign_test_items_index()
        if not self._test_unseen_users:
            self._assign_test_users_index()
        else:
            self._reindex_test_users()

    def _assign_test_items_index(self):
        itemid = self.fields.itemid
        self._map_entity(itemid, self._test.testset)
        self._map_entity(itemid, self._test.evalset)

    def _assign_test_users_index(self):
        userid = self.fields.userid
        self._map_entity(userid, self._test.testset)
        self._map_entity(userid, self._test.evalset)

    def _reindex_test_users(self):
        self._reindex_testset_users()
        self._assign_holdout_users_index()

    def _filter_short_sessions(self):
        userid = self.fields.userid
        holdout = self._test.evalset

        holdout_sessions = holdout.groupby(userid, sort=False)
        holdout_sessions_len = holdout_sessions.size()

        invalid_sessions = (holdout_sessions_len!=self.holdout_size)
        if invalid_sessions.any():
            n_invalid_sessions = invalid_sessions.sum()
            invalid_session_index = invalid_sessions.index[invalid_sessions]
            holdout.query('{} not in @invalid_session_index'.format(userid), inplace=True)
            if self.verbose:
                msg = '{} of {} {}\'s were filtered out from holdout. Reason: not enough items.'
                print msg.format(n_invalid_sessions, len(invalid_sessions), userid)

    def _align_test_users(self):
        if self._test.testset is None:
            return

        userid = self.fields.userid
        testset = self._test.testset
        holdout = self._test.evalset

        holdout_in_testset = holdout[userid].isin(testset[userid].unique())
        testset_in_holdout = testset[userid].isin(holdout[userid].unique())

        if not holdout_in_testset.all():
            invalid_holdout_users = holdout.loc[~holdout_in_testset, userid]
            n_unique_users = invalid_holdout_users.nunique()
            holdout.drop(invalid_holdout_users.index, inplace=True)
            if self.verbose:
                REASON = 'Reason: inconsistent with testset'
                msg = '{} {}\'s were filtered out from holdout. {}.'
                print msg.format(n_unique_users, userid, REASON)

        if not testset_in_holdout.all():
            invalid_testset_users = testset.loc[~testset_in_holdout, userid]
            n_unique_users = invalid_testset_users.nunique()
            testset.drop(invalid_testset_users.index, inplace=True)
            if self.verbose:
                REASON = 'Reason: inconsistent with holdout'
                msg = '{} {}\'s were filtered out from testset. {}.'
                print msg.format(n_unique_users, userid, REASON)

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

    def _map_entity(self, entity, dataset):
        if dataset is None:
            return

        entity_type = self.fields._fields[self.fields.index(entity)]
        index_data = getattr(self.index, entity_type)

        if index_data is None:
            return

        try:
            seen_entities_index = index_data.training
        except AttributeError:
            seen_entities_index = index_data

        entity_index_map = seen_entities_index.set_index('old').new
        dataset.loc[:, entity] = dataset.loc[:, entity].map(entity_index_map)

    def _filter_unseen_entity(self, entity, dataset, label):
        if dataset is None:
            return

        entity_type = self.fields._fields[self.fields.index(entity)]
        index_data = getattr(self.index, entity_type)

        if index_data is None:
            # TODO factorize training or get unique values
            raise NotImplementedError

        try:
            seen_entities = index_data.training['old']
        except AttributeError:
            seen_entities = index_data['old']

        seen_data = dataset[entity].isin(seen_entities)
        if not seen_data.all():
            n_unseen_entities = dataset.loc[~seen_data, entity].nunique()
            dataset.query('{} in @seen_entities'.format(entity), inplace=True)
            #unseen_index = dataset.index[unseen_entities]
            #dataset.drop(unseen_index, inplace=True)
            if self.verbose:
                UNSEEN = 'not in the training data'
                msg = '{} unique {}\'s within {} {} interactions were filtered. Reason: {}.'
                print msg.format(n_unseen_entities, entity, (~seen_data).sum(), label, UNSEEN)

    def _reindex_testset_users(self):
        userid = self.fields.userid
        user_index = self.reindex(self._test.testset, userid, sort=False)
        self.index = self.index._replace(userid=self.index.userid._replace(test=user_index))

    def _assign_holdout_users_index(self):
        # this is only for state 4
        userid = self.fields.userid
        test_user_index = self.index.userid.test.set_index('old').new
        self._test.evalset.loc[:, userid] = self._test.evalset.loc[:, userid].map(test_user_index)

    def _try_sort_test_data(self):
        userid = self.fields.userid
        testset = self._test.testset
        holdout = self._test.evalset
        if testset is not None:
            testset.sort_values(userid, inplace=True)
        if holdout is not None:
            holdout.sort_values(userid, inplace=True)


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


    def _sample_holdout(self, test_split):
        userid = self.fields.userid
        # TODO order_field may also change - need to check it as well
        order_field = self._custom_order or self.fields.feedback

        selector = self._data.loc[test_split, order_field]
        # data may have many items with the same top ratings
        # randomizing the data helps to avoid biases in that case
        if self._permute_tops and not self._random_holdout:
            random_state = np.random.RandomState(self.seed)
            selector = selector.sample(frac=1, random_state=random_state)

        grouper = selector.groupby(self._data[userid], sort=False)

        if self._random_holdout: #randomly sample data for evaluation
            random_state = np.random.RandomState(self.seed)
            if self._holdout_size >= 1:
                holdout = grouper.apply(random_choice, self._holdout_size, random_state)
            else:
                holdout = grouper.apply(random_sample, self._holdout_size, random_state)
        elif self._negative_prediction: #try to holdout negative only examples
            if self._holdout_size >= 1:
                holdout = grouper.nsmallest(self._holdout_size, keep='last')
            else:
                raise NotImplementedError
        else: #standard top-score prediction mode
            if self._holdout_size >= 1:
                holdout = grouper.nlargest(self._holdout_size, keep='last')
            else:
                raise NotImplementedError

        holdout_index = holdout.index.get_level_values(1)
        return self._data.loc[holdout_index]


    def _sample_testset(self, test_split, holdout_index):
        data = self._data[test_split].drop(holdout_index)

        test_sample = self._test_sample
        if not test_sample:
            return data

        userid = self.fields.userid
        if test_sample > 0: # sample at most test_sample items
            random_state = np.random.RandomState(self.seed)
            sampled = (data.groupby(userid, sort=False, group_keys=False)
                            .apply(random_choice, test_sample, random_state))
        else: # sample at most test_sample items with the worst feedback from user
            feedback = self.fields.feedback
            idx = (data.groupby(userid, sort=False)[feedback]
                        .nsmallest(-test_sample).index.get_level_values(1))
            sampled = data.loc[idx]
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
        testset = self.test.testset
        holdout = self.test.evalset

        if testset is None:
            if self._test_unseen_users or (holdout is None):
                raise ValueError('Unable to read test data')
            userid = self.fields.userid
            test_users = holdout[userid].drop_duplicates()

            if self.index.userid.training.new.isin(test_users).all():
                testset = self.training
            else:
                testset = (self.training.query('{} in @test_users'.format(userid))
                                 .sort_values(userid))

            self._test = self._test._replace(testset=testset)

        user_idx = testset[userid].values.astype(np.intp)
        item_idx = testset[itemid].values.astype(np.intp)
        fdbk_val = testset[feedback].values

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
        num_users = self.test.evalset[userid].nunique()
        try:
            item_index = self.index.itemid.training
        except AttributeError:
            item_index = self.index.itemid
        num_items = item_index.shape[0]
        shape = (num_users, num_items)

        if tensor_mode:
            num_fdbks = self.index.feedback.shape[0]
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
