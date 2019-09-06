import numpy as np
import pandas as pd
from scipy.sparse import issparse

from polara.recommender.data import RecommenderData


class SideRelationsMixin:
    def __init__(self, *args, relations_matrices, relations_indices, **kwargs):
        super().__init__(*args, **kwargs)

        entities = [self.fields.userid, self.fields.itemid]
        self._rel_idx = {entity: pd.Series(index=idx, data=np.arange(len(idx)), copy=False)
                                 if idx is not None else None
                         for entity, idx in relations_indices.items()
                         if entity in entities}
        self._rel_mat = {entity: mat for entity, mat in relations_matrices.items() if entity in entities}
        self._relations = dict.fromkeys(entities)

        self.subscribe(self.on_change_event, self._clean_relations)

    def _clean_relations(self):
        self._relations = dict.fromkeys(self._relations.keys())

    @property
    def item_relations(self):
        entity = self.fields.itemid
        return self.get_relations_matrix(entity)

    @property
    def user_relations(self):
        entity = self.fields.userid
        return self.get_relations_matrix(entity)

    def get_relations_matrix(self, entity):
        relations = self._relations.get(entity, None)
        if relations is None:
            self._update_relations(entity)
        return self._relations[entity]

    def _update_relations(self, entity):
        rel_mat = self._rel_mat[entity]
        if rel_mat is None:
            self._relations[entity] = None
        else:
            if self.verbose:
                print(f'Updating {entity} relations matrix')

            index_data = self.get_entity_index(entity)
            entity_idx = index_data['old']

            rel_idx = entity_idx.map(self._rel_idx[entity]).values
            rel_mat = self._rel_mat[entity][:, rel_idx][rel_idx, :]

            self._relations[entity] = rel_mat


class IdentityDiagonalMixin:
    def _update_relations(self, *args, **kwargs):
        super()._update_relations(*args, **kwargs)
        for rel_mat in self._relations.values():
            if rel_mat is not None:
                if issparse(rel_mat):
                    rel_mat.setdiag(1)
                else:
                    np.fill_diagonal(rel_mat, 1)


class SimilarityDataModel(IdentityDiagonalMixin, SideRelationsMixin, RecommenderData): pass
