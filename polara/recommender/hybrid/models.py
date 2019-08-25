import math
import scipy as sp
import numpy as np
from scipy.sparse.linalg import LinearOperator
from string import Template

try:
    from sksparse import __version__ as sk_sp_version
except ImportError:
    SPARSE_MODE = False
else:
    assert sk_sp_version >= '0.4.3'
    SPARSE_MODE = True
    from sksparse.cholmod import cholesky as cholesky_decomp_sparse

from polara.recommender.models import RecommenderModel, ProbabilisticMF, ScaledMatrixMixin, SVDModel
from polara.lib.optimize import kernelized_pmf_sgd, local_collective_embeddings
from polara.lib.sparse import sparse_dot
from polara.lib.similarity import stack_features
from polara.tools.timing import track_time
from polara.lib.cholesky import CholeskyFactor



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


    def build_item_graph(self, item_features, n_neighbors):
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise NotImplementedError('Install scikit-learn to construct graph for LCE model.')
        else:
            nbrs = NearestNeighbors(n_neighbors=1 + n_neighbors).fit(item_features)
        if self.binary_features:
            return nbrs.kneighbors_graph(item_features)
        return nbrs.kneighbors_graph(item_features, mode='distance')


    def build(self):
        # prepare input matrix for learning the model
        Xs, lbls = stack_features(self.item_data, normalize=False) # item-features sparse matrix
        Xu = self.get_training_matrix().T # item-user sparse matrix

        n_nbrs = min(self.max_neighbours, int(math.sqrt(Xs.shape[0])))
        A = self.build_item_graph(Xs, n_nbrs)

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


class CholeskyFactorsMixin:
    def __init__(self, *args, **kwargs):
        self._sparse_mode = SPARSE_MODE
        super().__init__(*args, **kwargs)
        entities = [self.data.fields.userid, self.data.fields.itemid]
        self._cholesky = dict.fromkeys(entities)

        self._features_weight = 0.5
        self.data.subscribe(self.data.on_change_event, self._clean_cholesky)

    def _clean_cholesky(self):
        self._cholesky = {entity:None for entity in self._cholesky.keys()}

    def _update_cholesky(self):
        for entity, cholesky in self._cholesky.items():
            if cholesky is not None:
                self._update_cholesky_inplace(entity)

    @property
    def features_weight(self):
        return self._features_weight

    @features_weight.setter
    def features_weight(self, new_val):
        if new_val != self._features_weight:
            self._features_weight = new_val
            self._update_cholesky()
            self._renew_model()

    @property
    def item_cholesky_factor(self):
        itemid = self.data.fields.itemid
        return self.get_cholesky_factor(itemid)

    @property
    def user_cholesky_factor(self):
        userid = self.data.fields.userid
        return self.get_cholesky_factor(userid)

    def get_cholesky_factor(self, entity):
        cholesky = self._cholesky.get(entity, None)
        if cholesky is None:
            self._update_cholesky_factor(entity)
        return self._cholesky[entity]

    def _update_cholesky_factor(self, entity):
        entity_similarity = self.data.get_relations_matrix(entity)
        if entity_similarity is None:
            self._cholesky[entity] = None
        else:
            if self._sparse_mode:
                cholesky_decomp = cholesky_decomp_sparse
                mode = 'sparse'
            else:
                raise NotImplementedError

            weight = self.features_weight
            beta = (1.0 - weight) / weight
            if self.verbose:
                print('Performing {} Cholesky decomposition for {} similarity'.format(mode, entity))

            msg = Template('Cholesky decomposition computation time: $time')
            with track_time(verbose=self.verbose, message=msg):
                self._cholesky[entity] = CholeskyFactor(cholesky_decomp(entity_similarity, beta=beta))

    def _update_cholesky_inplace(self, entity):
        entity_similarity = self.data.get_relations_matrix(entity)
        if self._sparse_mode:
            weight = self.features_weight
            beta = (1.0 - weight) / weight
            if self.verbose:
                print('Updating Cholesky decomposition inplace for {} similarity'.format(entity))

            msg = Template('    Cholesky decomposition update time: $time')
            with track_time(verbose=self.verbose, message=msg):
                self._cholesky[entity].update_inplace(entity_similarity, beta)
        else:
            raise NotImplementedError

    def build_item_projector(self, v):
        cholesky_items = self.item_cholesky_factor
        if cholesky_items is not None:
            if self.verbose:
                print(f'Building {self.data.fields.itemid} projector for {self.method}')
            msg = Template('    Solving triangular system: $time')
            with track_time(verbose=self.verbose, message=msg):
                self.factors['items_projector_left'] = cholesky_items.T.solve(v)
            msg = Template('    Applying Cholesky factor: $time')
            with track_time(verbose=self.verbose, message=msg):
                self.factors['items_projector_right'] = cholesky_items.dot(v)

    def get_item_projector(self):
        vl = self.factors.get('items_projector_left', None)
        vr = self.factors.get('items_projector_right', None)
        return vl, vr


class HybridSVD(CholeskyFactorsMixin, SVDModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = 'HybridSVD'
        self.precompute_auxiliary_matrix = False

    def _check_reduced_rank(self, rank):
        super()._check_reduced_rank(rank)
        self.round_item_projector(rank)

    def round_item_projector(self, rank):
        vl, vr = self.get_item_projector()
        if (vl is not None) and (rank < vl.shape[1]):
            self.factors['items_projector_left'] = vl[:, :rank]
            self.factors['items_projector_right'] = vr[:, :rank]

    def build(self, *args, **kwargs):
        if not self._sparse_mode:
            raise NotImplementedError('Check the installation of scikit-sparse package.')

        # the order matters - trigger on_change events first
        svd_matrix = self.get_training_matrix(dtype=np.float64)
        cholesky_items = self.item_cholesky_factor
        cholesky_users = self.user_cholesky_factor

        if self.precompute_auxiliary_matrix:
            if cholesky_items is not None:
                svd_matrix = cholesky_items.T.dot(svd_matrix.T).T
                cholesky_items._L = None
            if cholesky_users is not None:
                svd_matrix = cholesky_users.T.dot(svd_matrix)
                cholesky_users._L = None
            operator = svd_matrix
        else:
            if cholesky_items is not None:
                L_item = cholesky_items
            else:
                L_item = sp.sparse.eye(svd_matrix.shape[1])
            if cholesky_users is not None:
                L_user = cholesky_users
            else:
                L_user = sp.sparse.eye(svd_matrix.shape[0])

            def matvec(v):
                return L_user.T.dot(svd_matrix.dot(L_item.dot(v)))
            def rmatvec(v):
                return L_item.T.dot(svd_matrix.T.dot(L_user.dot(v)))
            operator = LinearOperator(svd_matrix.shape, matvec, rmatvec)

        super().build(*args, operator=operator, **kwargs)
        self.build_item_projector(self.factors[self.data.fields.itemid])

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        vl, vr = self.get_item_projector()
        scores = test_matrix.dot(vr).dot(vl.T)
        return scores, slice_data


class ScaledHybridSVD(ScaledMatrixMixin, HybridSVD): pass
