import scipy as sp
import numpy as np

from polara.recommender.models import RecommenderModel, ProbabilisticMF
from polara.lib.optimize import kernelized_pmf_sgd, local_collective_embeddings
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


class LCEModel(RecommenderModel):
    def __init__(self, *args, item_features=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank = 10
        self.factors = {}
        self.alpha = 0.1
        self.beta = 0.05
        self.max_neighbours = 10
        self.item_features = item_features
        self.binary_features = True
        self._item_data = None
        self.feature_labels = None
        self.seed = None
        self.show_error = False
        self.regularization = 1
        self.max_iterations = 15
        self.tolerance = 0.0001
        self.method = 'LCE'
        self.data.subscribe(self.data.on_change_event, self._clean_metadata)

    def _clean_metadata(self):
        self._item_data = None
        self.feature_labels = None

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, new_value):
        if new_value != self._rank:
            self._rank = new_value
            self._is_ready = False
            self._recommendations = None

    @property
    def item_data(self):
        if self.item_features is not None:
            if self._item_data is None:
                index_data = getattr(self.data.index, 'itemid')

                try:
                    item_index = index_data.training
                except AttributeError:
                    item_index = index_data

                self._item_data = self.item_features.reindex(item_index.old.values, # make correct sorting
                                                            fill_value=[])
        else:
            self._item_data = None
        return self._item_data


    def build(self):
        # prepare input matrix for learning the model
        Xs, lbls = stack_features(self.item_data, normalize=False) # item-features sparse matrix
        Xu = self.get_training_matrix().T # item-user sparse matrix

        n_nbrs = min(self.max_neighbours, int(math.sqrt(Xs.shape[0])))
        A = construct_A(Xs, n_nbrs, binary=self.binary_features)

        with track_time(self.training_time, verbose=self.verbose, model=self.method):
            W, Hu, Hs = local_collective_embeddings(Xs, Xu, A,
                                                    k=self.rank,
                                                    alpha=self.alpha,
                                                    beta=self.beta,
                                                    lamb=self.regularization,
                                                    epsilon=self.tolerance,
                                                    maxiter=self.max_iterations,
                                                    seed=self.seed,
                                                    verbose=self.show_error)

        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        self.factors[userid] = Hu.T
        self.factors[itemid] = W
        self.factors['item_features'] = Hs.T
        self.feature_labels = lbls

    def get_recommendations(self):
        if self.data.warm_start:
            raise NotImplementedError
        else:
            return super().get_recommendations()

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        slice_data = self._slice_test_data(test_data, start, stop)

        user_factors = self.factors[userid][test_users[start:stop], :]
        item_factors = self.factors[itemid]
        scores = user_factors.dot(item_factors.T)
        return scores, slice_data
