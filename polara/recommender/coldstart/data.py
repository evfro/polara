from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from polara.lib.similarity import build_indicator_matrix
from polara.recommender.data import RecommenderData


class ItemColdStartData(RecommenderData):
    def __init__(self, *args, **kwargs):
        self.meta_data = kwargs.pop('meta_data', None)
        super(ItemColdStartData, self).__init__(*args, **kwargs)

        self._test_ratio = 0.2
        self._warm_start = False
        self._holdout_size = -1  # needed for correct processing of test data

        # build unique items list to split them by folds
        itemid = self.fields.itemid
        permute = np.random.RandomState(self.seed).permutation
        self._unique_items = permute(self._data[itemid].unique())

        self._test_sample = None  # fraction of representative users from train
        self._repr_users = None

    @property
    def representative_users(self):
        if self._repr_users is None:
            sample = self.test_sample
            if sample:
                sample_params = {'frac' if sample < 1 else 'n': sample,
                                 'random_state': np.random.RandomState(self.seed)}
                all_users = self.index.userid.training
                self._repr_users = all_users.sample(**sample_params).sort_values('new')
        return self._repr_users

    def prepare(self):
        super(ItemColdStartData, self).prepare()

        if any(self._last_update_rule.values()):
            self._post_process_cold_items()

    def _split_test_index(self):
        itemid = self.fields.itemid

        item_idx = np.arange(len(self._unique_items))
        cold_items_split = self._split_fold_index(item_idx, len(item_idx), self._test_fold, self._test_ratio)

        cold_items = self._unique_items[cold_items_split]
        cold_items_mask = self._data[itemid].isin(cold_items)
        return cold_items_mask

    def _check_state_transition(self):
        assert not self._warm_start
        assert self._holdout_size != 0  # needed for correct processing of test data
        assert self._test_ratio > 0
        new_state, update_rule = super(ItemColdStartData, self)._check_state_transition()

        # handle change of test_sample value which is not handled
        # in standard state 3 scenario (as there's no testset)
        if '_test_sample' in self._change_properties:
            update_rule['test_update'] = True
            self._repr_users = None
        return new_state, update_rule

    def _sample_holdout(self, test_split, group_id=None):
        itemid = self.fields.itemid

        if self._holdout_size > 0:
            holdout = super(ItemColdStartData, self)._sample_holdout(test_split, group_id=itemid)
        else:
            holdout = self._data.loc[test_split, list(self.fields)]

        itemid_cold = '{}_cold'.format(itemid)
        return holdout.rename(columns={itemid: itemid_cold}, copy=False)

    def _try_drop_unseen_test_items(self):
        # there will be no such items except cold-start items
        pass

    def _filter_short_sessions(self, group_id=None):
        group_id = '{}_cold'.format(self.fields.itemid)
        super(ItemColdStartData, self)._filter_short_sessions(group_id=group_id)

    def _assign_test_items_index(self):
        if self.build_index:
            self._reindex_cold_items()

    def _reindex_cold_items(self):
        itemid_cold = '{}_cold'.format(self.fields.itemid)
        cold_item_index = self.reindex(self._test.holdout, itemid_cold, inplace=True, sort=False)

        try:  # check if already modified item index to avoid nested assignemnt
            item_index = self.index.itemid.training
        except AttributeError:
            item_index = self.index.itemid

        new_item_index = (namedtuple('ItemIndex', 'training cold_start')
                          ._make([item_index, cold_item_index]))
        self.index = self.index._replace(itemid=new_item_index)

    def _try_sort_test_data(self):
        # no need to sort by users
        pass

    def _post_process_cold_items(self):
        self._clean_representative_users()
        self._verify_cold_items_representatives()
        self._verify_cold_items_features()
        self._try_cleanup_cold_items()
        self._sort_by_cold_items()

    def _clean_representative_users(self):
        # TODO don't clean if training data and test_sample value are not changed
        self._repr_users = None

    def _verify_cold_items_representatives(self):
        repr_users = self.representative_users
        if repr_users is None:
            return

        # post-processing to leave only representative users
        # potentially filters out some cold items as well
        userid = self.fields.userid
        holdout = self._test.holdout  # use _ to avoid recusrion
        is_repr_user = holdout[userid].isin(repr_users.new)

        itemid_cold = '{}_cold'.format(self.fields.itemid)
        repr_items = holdout.loc[is_repr_user, itemid_cold].unique()
        item_index = self.index.itemid
        is_repr_item = item_index.cold_start.new.isin(repr_items)
        if not is_repr_item.all():
            item_index.cold_start['is_repr'] = is_repr_item

    def _verify_cold_items_features(self):
        if self.meta_data is None:
            return

        if self.meta_data.shape[1] > 1:
            try:  # agg is supported only starting from pandas v.0.20.0
                features_melted = self.meta_data.agg('sum', axis=1)
            except AttributeError:  # fall back to much slower but more general option
                features_melted = (self.meta_data.apply(lambda x: [x.sum()], axis=1)
                                                 .apply(lambda x: x[0]))
        else:
            features_melted = self.meta_data.iloc[:, 0]

        feature_labels = defaultdict(lambda: len(feature_labels))
        labels = features_melted.apply(lambda x: [feature_labels[i] for i in x])

        item_index = self.index.itemid
        cold_idx = item_index.cold_start.old
        seen_idx = item_index.training.old

        max_items = len(feature_labels)
        cold_items_matrix = build_indicator_matrix(labels.loc[cold_idx], max_items)
        seen_items_matrix = build_indicator_matrix(labels.loc[seen_idx], max_items)
        # valid -> has at least 1 feature intersecting with any of seen items
        is_valid_item = cold_items_matrix.dot(seen_items_matrix.T).getnnz(axis=1) > 0
        if not is_valid_item.all():
            item_index.cold_start['is_valid'] = is_valid_item

    def _try_cleanup_cold_items(self):
        holdout = self._test.holdout
        cold_index = self.index.itemid.cold_start
        item_index_query = []
        holdout_query = []

        if 'is_valid' in cold_index:
            item_index_query.append('is_valid')
            itemid_cold = '{}_cold'.format(self.fields.itemid)
            holdout_query.append('{} in @cold_index.new'.format(itemid_cold))

        if 'is_repr' in cold_index:
            item_index_query.append('is_repr')
            repr_users = self.representative_users.new
            holdout_query.append('{} in @repr_users'.format(self.fields.userid))

        item_index_query = ' and '.join([q for q in item_index_query if q])
        if item_index_query:
            cold_index.query(item_index_query, inplace=True)

        holdout_query = ' and '.join([q for q in holdout_query if q])
        if holdout_query:
            holdout.query(holdout_query, inplace=True)

    def _sort_by_cold_items(self):
        itemid_cold = '{}_cold'.format(self.fields.itemid)
        cold_index = self.index.itemid.cold_start
        cold_index.sort_values('new', inplace=True)
        holdout = self._test.holdout
        holdout.sort_values(itemid_cold, inplace=True)


class FeatureSimilarityMixin(object):
    def __init__(self, sim_mat, sim_idx, *args, **kwargs):
        super(FeatureSimilarityMixin, self).__init__(*args, **kwargs)

        entities = [self.fields.userid, self.fields.itemid]
        self._sim_idx = {entity: pd.Series(index=idx, data=np.arange(len(idx)), copy=False)
                         if idx is not None else None
                         for entity, idx in sim_idx.items()
                         if entity in entities}
        self._sim_mat = {entity: mat for entity, mat in sim_mat.items() if entity in entities}
        self._similarity = dict.fromkeys(entities)

        self.subscribe(self.on_change_event, self._clean_similarity)

    def _clean_similarity(self):
        self._similarity = dict.fromkeys(self._similarity.keys())

    @property
    def item_similarity(self):
        entity = self.fields.itemid
        return self.get_similarity_matrix(entity)

    @property
    def user_similarity(self):
        entity = self.fields.userid
        return self.get_similarity_matrix(entity)

    def get_similarity_matrix(self, entity):
        similarity = self._similarity.get(entity, None)
        if similarity is None:
            self._update_similarity(entity)
        return self._similarity[entity]

    def _update_similarity(self, entity):
        sim_mat = self._sim_mat[entity]
        if sim_mat is None:
            self._similarity[entity] = None
        else:
            if self.verbose:
                print('Updating {} similarity matrix'.format(entity))

            entity_type = self.fields._fields[self.fields.index(entity)]
            index_data = getattr(self.index, entity_type)

            try:  # check whether custom index is introduced
                entity_idx = index_data.training['old']
            except AttributeError:  # fall back to standard case
                entity_idx = index_data['old']

            sim_idx = entity_idx.map(self._sim_idx[entity]).values
            sim_mat = self._sim_mat[entity][:, sim_idx][sim_idx, :]

            if issparse(sim_mat):
                sim_mat.setdiag(1)
            else:
                np.fill_diagonal(sim_mat, 1)
            self._similarity[entity] = sim_mat


class ColdSimilarityMixin(object):
    @property
    def cold_items_similarity(self):
        itemid = self.fields.itemid
        return self.get_cold_similarity(itemid)

    @property
    def cold_users_similarity(self):
        userid = self.fields.userid
        return self.get_cold_similarity(userid)

    def get_cold_similarity(self, entity):
        sim_mat = self._sim_mat[entity]

        if sim_mat is None:
            return None

        fields = self.fields
        entity_type = fields._fields[fields.index(entity)]
        index_data = getattr(self.index, entity_type)

        similarity_index = self._sim_idx[entity]
        seen_idx = index_data.training['old'].map(similarity_index).values
        cold_idx = index_data.cold_start['old'].map(similarity_index).values

        return sim_mat[:, seen_idx][cold_idx, :]


class ColdStartSimilarityDataModel(ColdSimilarityMixin, FeatureSimilarityMixin, ItemColdStartData):
    pass
