from collections import defaultdict
from collections import namedtuple
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd

from polara.lib.sparse import inverse_permutation
from polara.recommender import defaults


def random_choice(df, num, random_state):
    n = df.shape[0]
    if n > num:
        return df.take(random_state.choice(n, num, replace=False), is_copy=False)
    else:
        return df


def random_sample(df, frac, random_state):
    return df.sample(frac=frac, random_state=random_state)


def group_largest_fraction(data, frac, groupid, by):
    def return_order(a):
        return np.take(range(1, len(a) + 1), inverse_permutation(np.argsort(a)))

    grouper = data.groupby(groupid, sort=False)[by]
    ordered = grouper.transform(return_order)
    largest = ordered.groupby(data[groupid], sort=False).transform(lambda x: x > round(frac * x.shape[0]))
    return largest


class EventNotifier:
    def __init__(self, events=None):
        self._subscribers = {}
        if events is not None:
            assert isinstance(events, list)
            for event in events:
                self.register_event(event)

    def register_event(self, event):
        self._subscribers[event] = WeakKeyDictionary({})

    def unregister_event(self, event):
        del self._subscribers[event]

    def _get_subscribers(self, event):
        return self._subscribers[event]

    def subscribe(self, event, callback):
        subscriber = callback.__self__
        func = callback.__func__
        self._get_subscribers(event).setdefault(subscriber, set()).add(func)

    def unsubscribe(self, event, subscriber):
        del self._get_subscribers(event)[subscriber]

    def unsubscribe_any(self, subscriber):
        for event in self._subscribers:
            subscribers = self._get_subscribers(event)
            if subscriber in subscribers:
                del subscribers[subscriber]

    def __call__(self, event):
        self._notify(event)

    def _notify(self, event):
        subscribers = self._get_subscribers(event)
        for subscriber_ref in subscribers.keyrefs():
            subscriber = subscriber_ref()
            if subscriber is not None:
                callbacks = subscribers.get(subscriber)
                for callback in list(callbacks):
                    callback(subscriber)


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
class RecommenderData:
    _std_fields = ('userid', 'itemid', 'feedback')

    _config = {'_shuffle_data', '_test_ratio', '_test_fold',
               '_warm_start', '_holdout_size', '_test_sample',
               '_permute_tops', '_random_holdout', '_negative_prediction'}

    def __init__(self, data, userid, itemid, feedback=None, custom_order=None, seed=None):
        self.name = None
        fields = [userid, itemid, feedback]

        if data is None:
            cols = fields + [custom_order]
            self._data = data = pd.DataFrame(columns=[c for c in cols if c])
        else:
            self._data = data

        if data.duplicated(subset=[f for f in fields if f]).any():
            # unstable in pandas v. 17.0, only works in <> v.17.0
            # rely on deduplicated data in many places - makes data processing more efficient
            raise NotImplementedError('Data has duplicate values')

        self._custom_order = custom_order
        self.fields = namedtuple('Fields', self._std_fields)
        self.fields = self.fields(**dict(zip(self._std_fields, fields)))
        self.index = namedtuple('DataIndex', self._std_fields)
        self.index = self.index._make([None] * len(self._std_fields))

        self._set_defaults()
        self._change_properties = set(['init'])  # container for changed properties
        # depending on config. For ex., test_fold - full_update,
        # TODO seed may also lead to either full_update or test_update
        # random_holdout - test_update. Need to implement checks
        # non-empty set is used to indicate non-initialized state ->
        # the data will be updated upon the first access of internal data splits
        self.seed = seed  # use with permute_tops, random_choice
        self.verify_sessions_length_distribution = True
        self.ensure_consistency = True  # drop test entities if not present in training
        self.build_index = True  # reindex data, avoid gaps in user and item index
        self._test_selector = None
        self._state = None  # None or 1 of {'_': 1, 'H': 11, '|': 2, 'd': 3, 'T': 4}
        self._last_update_rule = None

        self.on_change_event = 'on_change'
        self.on_update_event = 'on_update'
        self._notify = EventNotifier([self.on_change_event, self.on_update_event])
        # on_change indicates whether full data has been changed -> rebuild model
        # on_update indicates whether only test data has been changed -> renew recommendations
        self.verbose = True

    def subscribe(self, event, model_callback):
        self._notify.subscribe(event, model_callback)

    def unsubscribe(self, event, model):
        self._notify.unsubscribe(event, model)

    def _set_defaults(self, params=None):
        # [1:] omits undersacores in properties names
        params = params or [prop[1:] for prop in self._config]
        config_vals = defaults.get_config(params)
        for name, value in config_vals.items():
            internal_name = '_{}'.format(name)
            setattr(self, internal_name, value)

    def get_configuration(self):
        # [1:] omits undersacores in properties names, i.e. uses external name
        # in that case it prints worning if change is pending
        config = {attr[1:]: getattr(self, attr[1:]) for attr in self._config}
        return config

    @property
    def test(self):
        self.update()
        return self._test

    @property
    def training(self):
        self.update()  # both _test and _training attributes appear simultaneously
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
            print('The value of {} might be not effective yet.'.format(data_property[1:]))
        return getattr(self, data_property)

    def update(self):
        if self._change_properties:
            self.prepare()

    def prepare(self):
        if self.verbose:
            print('Preparing data...')

        update_rule = self._split_data()

        if update_rule['full_update']:
            self._try_reindex_training_data()

        if update_rule['full_update'] or update_rule['test_update']:
            self._try_drop_unseen_test_items()  # unseen = not present in training data
            self._try_drop_unseen_test_users()  # unseen = not present in training data
            self._try_drop_invalid_test_users()  # with too few items and/or if inconsistent between testset and holdout
            self._try_reindex_test_data()  # either assign known index, or (if testing for unseen users) reindex
            self._try_sort_test_data()

        if self.verbose:
            num_train_events = self.training.shape[0] if self.training is not None else 0
            num_holdout_events = self.test.holdout.shape[0] if self.test.holdout is not None else 0
            stats_msg = 'Done.\nThere are {} events in the training and {} events in the holdout.'
            print(stats_msg.format(num_train_events, num_holdout_events))

    def prepare_training_only(self):
        self.holdout_size = 0  # do not form holdout
        self.test_ratio = 0  # do not form testset
        self.warm_start = False  # required for correct state transition handling
        self.prepare()

    def _validate_config(self):
        if self._warm_start and not (self._holdout_size and self._test_ratio):
            raise ValueError('Both holdout_size and test_ratio must be positive when warm_start is set to True')
        if not self._warm_start and (self._holdout_size == 0) and (self._test_ratio > 0):
            raise ValueError('test_ratio cannot be nonzero when holdout_size is 0 and warm_start is set to False')

        assert self._test_ratio < 1, 'Value of test_ratio can\'t be greater than or equal to 1'

        if self._test_ratio:
            max_fold = 1.0 / self._test_ratio
            if self._test_fold > max_fold:
                raise ValueError('Test fold value cannot be greater than {}'.format(max_fold))

    def _check_state_transition(self):
        test_ratio_change = '_test_ratio' in self._change_properties
        test_fold_change = '_test_fold' in self._change_properties
        test_sample_change = '_test_sample' in self._change_properties
        test_data_change = test_fold_change or test_ratio_change
        holdout_sz_change = '_holdout_size' in self._change_properties
        unseen_usr_change = '_warm_start' in self._change_properties
        permute_change = '_permute_tops' in self._change_properties
        negative_change = ('_negative_prediction' in self._change_properties) and not self._random_holdout
        rnd_holdout_change = '_random_holdout' in self._change_properties
        any_holdout_change = holdout_sz_change or rnd_holdout_change or negative_change or permute_change
        empty_holdout = self._holdout_size == 0
        empty_testset = self._test_ratio == 0
        test_unseen = self._warm_start
        last_state = self._state
        update_rule = defaultdict(bool)
        new_state = last_state

        if unseen_usr_change:  # unseen_test_users is reserved for state 4 only!
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
        else:  # this assumes that warm_start is consistent with current state!
            if last_state == 1:  # hsz = 0, trt = 0, usn = False
                if holdout_sz_change:  # hsz > 0
                    new_state = 3 if test_ratio_change else 2
                    update_rule['full_update'] = True
                elif test_ratio_change:  # hsz = 0,  trt > 0
                    new_state = 11
                    update_rule['full_update'] = True

            elif last_state == 11:  # hsz = 0, trt > 0, usn = False
                if holdout_sz_change:  # hsz > 0
                    new_state = 2 if empty_testset else 3
                    update_rule['full_update'] = True
                elif test_data_change:  # hsz = 0
                    if empty_testset:  # hsz = 0, trt = 0
                        new_state = 1
                    update_rule['full_update'] = True

            elif last_state == 2:  # hsz > 0, trt = 0, usn = False
                if test_ratio_change:  # trt > 0
                    new_state = 11 if empty_holdout else 3
                    update_rule['full_update'] = True

                elif any_holdout_change:  # trt = 0
                    if empty_holdout:  # hsz = 0
                        new_state = 1
                    update_rule['full_update'] = True

            elif last_state == 3:  # hsz > 0, trt > 0, usn = False
                if test_data_change or any_holdout_change:
                    if empty_holdout:
                        new_state = 1 if empty_testset else 11
                    elif empty_testset:  # hsz > 0, trt = 0
                        new_state = 2
                    update_rule['full_update'] = True

            elif last_state == 4:  # hsz > 0, trt > 0, usn = True
                if any_holdout_change:
                    if empty_holdout:
                        if test_data_change:
                            new_state = 1 if empty_testset else 11
                            update_rule['full_update'] = True
                        else:  # hsz = 0, trt > 0
                            new_state = 11
                            update_rule['test_update'] = True
                    else:  # hsz > 0
                        if test_data_change:
                            if empty_testset:  # hsz > 0, trt = 0
                                new_state = 2
                            update_rule['full_update'] = True
                        else:  # including test_sample_change
                            update_rule['test_update'] = True
                else:  # hsz > 0
                    if test_data_change:
                        if empty_testset:  # hsz > 0, trt = 0
                            new_state = 2
                        update_rule['full_update'] = True
                    elif test_sample_change:
                        update_rule['test_update'] = True

            else:  # initial state
                if empty_holdout:
                    new_state = 1 if empty_testset else 11
                else:
                    if empty_testset:  # hsz > 0, trt = 0
                        new_state = 2
                    else:  # hsz > 0, trt > 0
                        new_state = 4 if test_unseen else 3
                update_rule['full_update'] = True

        return new_state, update_rule

    def _split_data(self):
        self._validate_config()
        new_state, update_rule = self._check_state_transition()

        full_update = update_rule['full_update']
        test_update = update_rule['test_update']

        if not (full_update or test_update):
            # TODO place assert new_state == self._state into tests
            if self.verbose:
                print('Data is ready. No action was taken.')
            return update_rule

        if self._test_ratio > 0:
            if full_update:
                test_split = self._split_test_index()
            else:  # test_update
                test_split = self._test_split
            if self._holdout_size == 0:  # state 11
                testset = holdout = None
                train_split = ~test_split
            else:  # state 3 or state 4
                # NOTE holdout_size = None is also here; this can be used in
                # subclasses like ItemColdStartData to preprocess data properly
                # in that case _sample_holdout must be modified accordingly
                holdout = self._sample_holdout(test_split)

                if self._warm_start:  # state 4
                    testset = self._sample_testset(test_split, holdout.index)
                    train_split = ~test_split
                else:  # state 3
                    testset = None  # will be computed if test data is requested
                    train_split = ~self._data.index.isin(holdout.index)
        else:  # test_ratio == 0
            testset = None  # will be computed if test data is requested
            test_split = slice(None)

            if self._holdout_size >= 1:  # state 2, sample holdout data per each user
                holdout = self._sample_holdout(test_split)
            elif self._holdout_size > 0:  # state 2, special case - sample whole data at once
                if self._random_holdout:
                    random_state = np.random.RandomState(self.seed)
                    holdout = self._data.sample(frac=self._holdout_size, random_state=random_state)
                else:
                    # TODO custom groupid support, not only userid
                    group_id = self.fields.userid
                    order_id = self._custom_order or self.fields.feedback
                    frac = self._holdout_size
                    largest = group_largest_fraction(self._data, frac, group_id, order_id)
                    holdout = self._data.loc[largest].copy()
            else:  # state 1
                holdout = None

            train_split = slice(None) if holdout is None else ~self._data.index.isin(holdout.index)

        self._state = new_state
        self._test_split = test_split
        self._test = namedtuple('TestData', 'testset holdout')._make([testset, holdout])

        if full_update:
            fields = [f for f in list(self.fields) if f is not None]
            if self._custom_order:
                fields.append(self._custom_order)
            self._training = self._data.loc[train_split, fields]
            self._notify(self.on_change_event)
        elif test_update:
            self._notify(self.on_update_event)

        self._last_update_rule = update_rule
        self._change_properties.clear()
        return update_rule

    def _split_test_index(self):
        sessions_size, sess_idx = self._get_sessions_info()
        n_sessions = len(sessions_size)
        test_split = self._split_fold_index(sess_idx, n_sessions, self._test_fold, self._test_ratio)
        return test_split

    def _get_sessions_info(self):
        userid = self.fields.userid
        user_sessions = self._data.groupby(userid, sort=True)  # KEEP TRUE HERE!
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
                print('Users are not uniformly ordered! Unable to split test set reliably.')
            self.verify_sessions_length_distribution = False
        user_sessions_len = user_sessions.size()
        return user_sessions_len, user_idx

    @staticmethod
    def is_not_uniform(idx, nbins=10, allowed_gap=0.75):
        idx_bins = pd.cut(idx, bins=nbins, labels=False)
        idx_bin_size = np.bincount(idx_bins)

        diff = idx_bin_size[:-1] - idx_bin_size[1:]
        monotonic = (diff < 0).all() or (diff > 0).all()
        huge_gap = (idx_bin_size.min() / idx_bin_size.max()) < allowed_gap
        return monotonic or huge_gap

    @staticmethod
    def _split_fold_index(idx, n_unique, fold, ratio):
        # supports both [0, 1, 2, 3] and [0, 0, 1, 1, 1, 2, 3, 3] types of idx
        # if idx contains only unique elements (1 case) then n_unique = len(idx)
        num = n_unique * ratio
        selection = (idx >= round((fold - 1) * num)) & (idx < round(fold * num))
        return selection

    def _try_reindex_training_data(self):
        if self.build_index:
            self._reindex_train_users()
            self._reindex_train_items()
            self._reindex_feedback()

    def _try_drop_unseen_test_items(self, mapping='old'):
        if self.ensure_consistency:
            itemid = self.fields.itemid
            self._filter_unseen_entity(itemid, self._test.testset, 'testset', mapping)
            self._filter_unseen_entity(itemid, self._test.holdout, 'holdout', mapping)

    def _try_drop_unseen_test_users(self, mapping='old'):
        if self.ensure_consistency and not self._warm_start:
            # even in state 3 there could be unseen users
            userid = self.fields.userid
            self._filter_unseen_entity(userid, self._test.holdout, 'holdout', mapping)

    def _try_drop_invalid_test_users(self):
        if self.holdout_size >= 1:
            # TODO remove that, when new evaluation arrives
            self._filter_short_sessions()  # ensure holdout conforms the holdout_size attribute
        self._align_test_users()  # ensure the same users are in both testset and holdout

    def _try_reindex_test_data(self):
        self._assign_test_items_index()
        if not self._warm_start:
            self._assign_test_users_index()
        else:
            self._reindex_test_users()

    def _assign_test_items_index(self):
        itemid = self.fields.itemid
        self._map_entity(itemid, self._test.testset)
        self._map_entity(itemid, self._test.holdout)

    def _assign_test_users_index(self):
        userid = self.fields.userid
        self._map_entity(userid, self._test.testset)
        self._map_entity(userid, self._test.holdout)

    def _reindex_test_users(self):
        self._reindex_testset_users()
        if self._test.holdout is not None:
            self._assign_holdout_users_index()

    def _filter_short_sessions(self, group_id=None):
        userid = self.fields.userid
        holdout = self._test.holdout
        group_id = group_id or userid

        holdout_sessions = holdout.groupby(group_id, sort=False)
        holdout_sessions_len = holdout_sessions.size()

        invalid_sessions = (holdout_sessions_len != self.holdout_size)
        if invalid_sessions.any():
            n_invalid_sessions = invalid_sessions.sum()
            invalid_session_index = invalid_sessions.index[invalid_sessions]
            holdout.query('{} not in @invalid_session_index'.format(group_id), inplace=True)
            if self.verbose:
                msg = '{} of {} {}\'s were filtered out from holdout. Reason: incompatible number of items.'
                print(msg.format(n_invalid_sessions, len(invalid_sessions), group_id))

    def _align_test_users(self):
        if (self._test.testset is None) or (self._test.holdout is None):
            return

        userid = self.fields.userid
        testset = self._test.testset
        holdout = self._test.holdout

        holdout_in_testset = holdout[userid].isin(testset[userid].unique())
        testset_in_holdout = testset[userid].isin(holdout[userid].unique())

        if not holdout_in_testset.all():
            invalid_holdout_users = holdout.loc[~holdout_in_testset, userid]
            n_unique_users = invalid_holdout_users.nunique()
            holdout.drop(invalid_holdout_users.index, inplace=True)
            if self.verbose:
                REASON = 'Reason: inconsistent with testset'
                msg = '{} {}\'s were filtered out from holdout. {}.'
                print(msg.format(n_unique_users, userid, REASON))

        if not testset_in_holdout.all():
            invalid_testset_users = testset.loc[~testset_in_holdout, userid]
            n_unique_users = invalid_testset_users.nunique()
            testset.drop(invalid_testset_users.index, inplace=True)
            if self.verbose:
                REASON = 'Reason: inconsistent with holdout'
                msg = '{} {}\'s were filtered out from testset. {}.'
                print(msg.format(n_unique_users, userid, REASON))

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

    def _filter_unseen_entity(self, entity, dataset, label, mapping):
        if dataset is None:
            return

        entity_type = self.fields._fields[self.fields.index(entity)]
        index_data = getattr(self.index, entity_type)

        if index_data is None:
            # TODO factorize training or get unique values
            raise NotImplementedError

        try:
            seen_entities = index_data.training[mapping]
        except AttributeError:
            seen_entities = index_data[mapping]

        seen_data = dataset[entity].isin(seen_entities)
        if not seen_data.all():
            n_unseen_entities = dataset.loc[~seen_data, entity].nunique()
            dataset.query('{} in @seen_entities'.format(entity), inplace=True)
            # unseen_index = dataset.index[unseen_entities]
            # dataset.drop(unseen_index, inplace=True)
            if self.verbose:
                UNSEEN = 'not in the training data'
                msg = '{} unique {}\'s within {} {} interactions were filtered. Reason: {}.'
                print(msg.format(n_unseen_entities, entity, (~seen_data).sum(), label, UNSEEN))

    def _reindex_testset_users(self):
        userid = self.fields.userid
        user_index = self.reindex(self._test.testset, userid, sort=False)
        self.index = self.index._replace(userid=self.index.userid._replace(test=user_index))

    def _assign_holdout_users_index(self):
        # this is only for state 4
        userid = self.fields.userid
        test_user_index = self.index.userid.test.set_index('old').new
        self._test.holdout.loc[:, userid] = self._test.holdout.loc[:, userid].map(test_user_index)

    def _try_sort_test_data(self):
        userid = self.fields.userid
        testset = self._test.testset
        holdout = self._test.holdout
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

    def _sample_holdout(self, test_split, group_id=None):
        # TODO order_field may also change - need to check it as well
        order_field = self._custom_order or self.fields.feedback or []

        selector = self._data.loc[test_split, order_field]
        # data may have many items with the same top ratings
        # randomizing the data helps to avoid biases in that case
        if self._permute_tops and not self._random_holdout:
            random_state = np.random.RandomState(self.seed)
            selector = selector.sample(frac=1, random_state=random_state)

        group_id = group_id or self.fields.userid
        grouper = selector.groupby(self._data[group_id], sort=False, group_keys=False)

        if self._random_holdout:  # randomly sample data for evaluation
            random_state = np.random.RandomState(self.seed)
            if self._holdout_size >= 1:  # pick at most _holdout_size elements
                holdout = grouper.apply(random_choice, self._holdout_size, random_state)
            else:
                holdout = grouper.apply(random_sample, self._holdout_size, random_state)
        elif self._negative_prediction:  # try to holdout negative only examples
            if self._holdout_size >= 1:  # pick at most _holdout_size elements
                holdout = grouper.nsmallest(self._holdout_size, keep='last')
            else:
                raise NotImplementedError
        else:  # standard top-score prediction mode
            if self._holdout_size >= 1:  # pick at most _holdout_size elements
                holdout = grouper.nlargest(self._holdout_size, keep='last')
            else:
                frac = self._holdout_size

                def sample_largest(x):
                    size = round(frac * len(x))
                    return x.iloc[np.argpartition(x, -size)[-size:]]
                holdout = grouper.apply(sample_largest)

        return self._data.loc[holdout.index]

    def _sample_testset(self, test_split, holdout_index):
        data = self._data[test_split].drop(holdout_index)

        test_sample = self._test_sample
        if not test_sample:
            return data

        userid = self.fields.userid
        if test_sample > 0:  # sample at most test_sample items
            random_state = np.random.RandomState(self.seed)
            sampled = (data.groupby(userid, sort=False, group_keys=False)
                           .apply(random_choice, test_sample, random_state))
        else:  # sample at most test_sample items with the worst feedback from user
            feedback = self.fields.feedback
            idx = (data.groupby(userid, sort=False)[feedback]
                       .nsmallest(-test_sample).index.get_level_values(1))
            sampled = data.loc[idx]
        return sampled

    @staticmethod
    def threshold_data(idx, val, threshold, filter_values=True):
        if threshold is None:
            return idx, val

        value_filter = val >= threshold
        if filter_values:
            val = val[value_filter]
            if isinstance(idx, tuple):
                idx = tuple([x[value_filter] for x in idx])
            else:
                idx = idx[value_filter, :]
        else:
            val[~value_filter] = 0
        return idx, val

    def to_coo(self, tensor_mode=False, feedback_threshold=None):
        userid, itemid, feedback = self.fields
        user_item_data = self.training[[userid, itemid]].values

        if tensor_mode:
            # TODO this recomputes feedback data every new functon call,
            # but if data has not changed - no need for this, make a property
            new_feedback, feedback_transform = self.reindex(self.training, feedback, inplace=False)
            self.index = self.index._replace(feedback=feedback_transform)

            idx = np.hstack((user_item_data, new_feedback[:, np.newaxis]))
            val = np.ones(self.training.shape[0],)
        else:
            idx = user_item_data
            if feedback is None:
                val = np.ones(self.training.shape[0],)
            else:
                val = self.training[feedback].values

        shp = tuple(idx.max(axis=0) + 1)
        idx, val = self.threshold_data(idx, val, feedback_threshold)
        idx = idx.astype(np.intp)
        val = np.ascontiguousarray(val)
        return idx, val, shp

    def _recover_testset(self, update_data=False):
        userid = self.fields.userid
        holdout = self.test.holdout
        test_users = holdout[userid].drop_duplicates()
        if self.index.userid.training.new.isin(test_users).all():
            testset = self.training
        else:
            testset = self.training.query('{} in @test_users'.format(userid))

        testset = testset.sort_values(userid)
        if update_data:
            self._test = self._test._replace(testset=testset)
        return testset

    def test_to_coo(self, tensor_mode=False, feedback_threshold=None):
        userid, itemid, feedback = self.fields
        testset = self.test.testset

        if testset is None:
            if self._warm_start or (self.test.holdout is None):
                raise ValueError('Unable to read test data')
            # returns already processed data, as it's based on the training set
            testset = self._recover_testset(update_data=False)

        user_idx = testset[userid].values.astype(np.intp)
        item_idx = testset[itemid].values.astype(np.intp)

        if tensor_mode:
            fdbk_val = testset[feedback]
            fdbk_idx = fdbk_val.map(self.index.feedback.set_index('old').new)
            if fdbk_idx.isnull().any():
                raise NotImplementedError('Not all values of feedback are present in training data')
            fdbk_idx = fdbk_idx.values.astype(np.intp)
            test_coo = (user_idx, item_idx, fdbk_idx)
        else:
            if feedback is None:
                fdbk_val = np.ones(testset.shape[0],)
            else:
                fdbk_val = testset[feedback].values
            test_coo = (user_idx, item_idx, fdbk_val)
        test_coo, val = self.threshold_data(test_coo[:-1], test_coo[-1], feedback_threshold, filter_values=False)
        return test_coo + (val,)

    def get_test_shape(self, tensor_mode=False):
        userid = self.fields.userid
        if self.test.holdout is None:
            num_users = self.test.testset[userid].nunique()
            # TODO make it a property
        else:
            num_users = self.test.holdout[userid].nunique()

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

    def set_test_data(self, testset=None, holdout=None, warm_start=False, test_users=None,
                      reindex=True, ensure_consistency=True, holdout_size=None, copy=True):
        '''Should be used only with custom data.'''
        if warm_start and ((testset is None) and (test_users is None)):
            raise ValueError('When warm_start is True, information about test users must be present. '
                             'Please provide either testset or test_users argument.')

        if (not warm_start) and (testset is not None):
            raise ValueError('When warm_start is False, testset argument shouldn\'t be used. '
                             'Make sure to provide at least one of holdout and test_users arguments instead.')

        if (test_users is not None) and (testset is not None):
            raise ValueError('testset and test_users cannot be provided together.')

        if copy:
            testset = testset.copy() if testset is not None else None
            holdout = holdout.copy() if holdout is not None else None

        if test_users is not None:
            fields = [f for f in list(self.fields) if f is not None]
            if self._custom_order:
                fields.append(self._custom_order)
            testset = self._data.loc[lambda x: x[self.fields.userid].isin(test_users), fields]

        self._test = namedtuple('TestData', 'testset holdout')._make([testset, holdout])
        self.index = self.index._replace(userid=self.index.userid._replace(test=None))

        self._warm_start = warm_start
        self._state = None
        self._last_update_rule = None
        self._test_ratio = -1
        self._holdout_size = holdout_size or -1
        self._notify(self.on_update_event)
        self._change_properties.clear()

        if (testset is None) and (holdout is None):
            return  # allows to cleanup data

        if ensure_consistency:  # allows to disable self.ensure_consistency without actually changing it
            index_mapping = 'old' if reindex else 'new'
            self._try_drop_unseen_test_items(mapping=index_mapping)  # unseen = not present in training data
            self._try_drop_unseen_test_users(mapping=index_mapping)  # unseen = not present in training data
        self._try_drop_invalid_test_users()  # inconsistent between testset and holdout
        if reindex:
            self._try_reindex_test_data()  # either assign known index, or reindex (if warm_start)
        self._try_sort_test_data()


class LongTailMixin:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        self.long_tail_holdout = kwargs.pop('long_tail_holdout', False)
        # use predefined list if defined
        self.short_head_items = kwargs.pop('short_head_items', None)
        # amount of feedback accumulated in short head
        self.head_feedback_frac = kwargs.pop('head_feedback_frac', 0.33)
        # fraction of popular items considered as short head
        self.head_items_frac = kwargs.pop('head_items_frac', None)
        self._long_tail_items = None
        super().__init__(*args, **kwargs)

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
            self.head_feedback_frac = None  # could in principle calculate real value instead
            items_frac = np.arange(1, len(popularity) + 1) / len(popularity)
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
        return super()._sample_holdout(data)

    def _sample_test_data(self, data):
        if self.long_tail_holdout:
            data = pd.concat([self.__head_data, data], copy=True)
            del self.__head_data
        return super()._sample_test_data(data)
