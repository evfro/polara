import scipy as sp
import numpy as np

from polara.recommender.models import RecommenderModel, ProbabilisticMF
from polara.lib.optimize import kernelized_pmf_sgd
from polara.lib.sparse import sparse_dot
from polara.tools.timing import track_time


class SimilarityAggregation(RecommenderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'SIM'
        self.implicit = False
        self.dense_output = False
        self.item_similarity_matrix = False

    def build(self):
        # use copy to prevent contaminating original data
        self.item_similarity_matrix = self.data.item_relations.copy()
        self.item_similarity_matrix.setdiag(0) # exclude self-links
        self.item_similarity_matrix.eliminate_zeros()

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        if self.implicit:
            test_matrix.data = np.ones_like(test_matrix.data)
        scores = sparse_dot(test_matrix, self.item_similarity_matrix, self.dense_output, True)
        return scores, slice_data


class KernelizedRecommenderMixin:
'''Based on the work:
Kernelized Probabilistic Matrix Factorization: Exploiting Graphs and Side Information
http://people.ee.duke.edu/~lcarin/kpmf_sdm_final.pdf
'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_type = 'reg'
        self.beta = 0.01
        self.gamma = 0.1
        self.sigma = 1
        self.kernel_update = None # will use default kernel update method
        self.sparse_kernel_format = True

        entities = [self.data.fields.userid, self.data.fields.itemid]
        self.factor_sigma = dict.fromkeys(entities, 1)
        self._kernel_matrices = dict.fromkeys(entities)

        self.data.subscribe(self.data.on_change_event, self._clean_kernel_data)

    def _compute_kernel(self, laplacian, kernel_type=None):
        kernel_type = kernel_type or self.kernel_type
        if kernel_type == 'dif': # diffusion
            return sp.sparse.linalg.expm(self.beta * laplacian) # dense matrix
        elif kernel_type == 'reg': # regularized laplacian
            n_entities = laplacian.shape[0]
            return sp.sparse.eye(n_entities).tocsr() + self.gamma * laplacian # sparse matrix
        else:
            raise ValueError

    def _update_kernel_matrices(self, entity):
        laplacian = self.data.get_relations_matrix(entity)
        if laplacian is None:
            sigma = self.factor_sigma[entity]
            n_entities = self.data.get_entity_index(entity).shape[0]
            kernel_matrix = (sigma**2) * sp.sparse.eye(n_entities).tocsr()
        else:
            kernel_matrix = self._compute_kernel(laplacian)
        self._kernel_matrices[entity] = kernel_matrix

    def _clean_kernel_data(self):
        self._kernel_matrices = dict.fromkeys(self._kernel_matrices.keys())

    @property
    def item_kernel_matrix(self):
        entity = self.data.fields.itemid
        return self.get_kernel_matrix(entity)

    @property
    def user_kernel_matrix(self):
        entity = self.data.fields.userid
        return self.get_kernel_matrix(entity)

    def get_kernel_matrix(self, entity):
        kernel_matrix = self._kernel_matrices.get(entity, None)
        if kernel_matrix is None:
            self._update_kernel_matrices(entity)
        return self._kernel_matrices[entity]


class KernelizedPMF(KernelizedRecommenderMixin, ProbabilisticMF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = kernelized_pmf_sgd
        self.method = 'KPMF'

    def build(self, *args, **kwargs):
        kernel_matrices = (self.user_kernel_matrix, self.item_kernel_matrix)
        kernel_config = dict(kernel_update=self.kernel_update,
                             sparse_kernel_format=self.sparse_kernel_format)
        super().build(kernel_matrices, *args, **kernel_config, **kwargs)
