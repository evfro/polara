from subprocess import call
from shlex import split
import sys
import pandas as pd
import numpy as np
from polara.recommender.models import SVDModel


# default settings
if sys.platform == 'win32':
    program = 'item_recommendation.exe'
else:
    program = 'item_recommendation'

num_factors = 10
learn_rate = 0.1
train_data_path = 'train_data.csv'
user_mapping = 'user_mapping'
item_mapping = 'item_mapping'
positive_only = True  # saves up space as MML anyway works with positive feedback only
#wrmf_weight = 1

command_template = (
    '{program}'
    ' --training-file={train_path}'
    ' --save-model={saved_model_path}'
    ' --recommender={algo}'
    ' --recommender-options={options}')


def join_path(*paths):
    return '/'.join([p.rstrip('/') for p in paths])


class MyMediaLiteWrapper(SVDModel):

    def __init__(self, library_path, data_folder, method_name, *args, **kwargs):
        super(MyMediaLiteWrapper, self).__init__(*args, **kwargs)
        self.path = library_path
        self.program = join_path(library_path, program)
        self.method = method_name
        self.name = method_name  # used to store model specific data
        self.data_folder = data_folder
        self.rank = num_factors
        self.learn_rate = learn_rate
        self.positive_only = positive_only
        self.orthogonal_factors = True
        self._items_biases = None

    @property
    def saved_model_path(self):
        return join_path(self.data_folder, self.name + '_model.txt')

    @property
    def train_data_path(self):
        return join_path(self.data_folder, '{}_{}'.format(self.data.name, train_data_path))

    @property
    def user_mapping_file(self):
        return join_path(self.data_folder, '{}_{}_{}.tsv'.format(self.name, self.data.name, user_mapping))

    @property
    def item_mapping_file(self):
        return join_path(self.data_folder, '{}_{}_{}.tsv'.format(self.name, self.data.name, item_mapping))

    @property
    def command(self):
        if self.positive_only:
            command = ('{} --save-user-mapping={{user_mapping}}'
                       ' --save-item-mapping={{item_mapping}}').format(command_template)
        else:
            command = ('{} --no-id-mapping'
                       ' --rating-threshold={{feedback_threshold}}').format(command_template)
        return command

    def _save_to_disk(self):
        if self.positive_only:
            feedback = self.data.fields.feedback
            pos_ind = self.data.training[feedback] >= self.feedback_threshold
            pos_data = self.data.training.loc[pos_ind]
            pos_data.to_csv(self.train_data_path, index=False, header=False)
        else:
            self.data.training.to_csv(self.train_data_path, index=False, header=False)

    def _run_external(self, debug=False):
        method_name = self.method.lower()
        if method_name.endswith('mf'):
            self.options = '"num_factors={num_factors}"'.format(num_factors=self.rank)
            # allow variations in naming - but the input to MML is strict
            if 'bpr' in method_name:
                method_name = 'BPRMF'
            if 'wr' in method_name:
                method_name = 'WRMF'
        else:
            raise NotImplementedError('Only matrix factorization methods are currently supported.')

        run_command = self.command.format(
            program=self.program,
            train_path=self.train_data_path,
            saved_model_path=self.saved_model_path,
            feedback_threshold=self.feedback_threshold,
            topk=self.topk,
            algo=method_name,
            options=self.options,
            user_mapping=self.user_mapping_file,
            item_mapping=self.item_mapping_file)

        return call(split(run_command)) if not debug else run_command

    @staticmethod
    def _remap_factors(entity_mapping, entity_factors, num_entities, num_factors):
        shape = (num_entities, num_factors)
        entity_id = np.repeat(entity_mapping.loc[:, 1].values, num_factors, axis=0).astype(np.int64)
        factor_id = entity_factors['col2'].values.astype(np.int64)
        entity_factors_idx = np.ravel_multi_index((entity_id, factor_id), dims=shape)
        entity_factors_new = np.zeros(shape)
        np.put(entity_factors_new, entity_factors_idx, entity_factors['col3'].values)
        return entity_factors_new

    def _parse_factors(self):
        model_data_path = self.saved_model_path
        model_params = pd.read_csv(model_data_path, skiprows=2, header=None,
                                   sep=' ', names=['col1', 'col2', 'col3'])
        num_users = self.data.index.userid.training.new.max() + 1
        num_items = self.data.index.itemid.new.max() + 1

        nu, nf = model_params.iloc[0, :2].astype(np.int64)
        boundary = nu * nf + 1
        ni = model_params.iloc[boundary, 0].astype(np.int64)

        users_factors = model_params.iloc[1:boundary, :]

        if model_params.shape[0] == ((nu + ni) * nf + 2):  # no biases
            items_biases = None
            items_factors = model_params.iloc[(boundary + 1):]
        elif model_params.shape[0] == ((nu + ni) * nf + ni + 3):  # has biases
            items_biases = model_params.iloc[(boundary + 1):(boundary + 1 + ni), 0].values
            items_factors = model_params.iloc[(boundary + 2 + ni):, :]
        else:
            NotImplementedError('{} data is not recognized.'.format(model_data_path))

        if self.positive_only:
            user_mapping = pd.read_csv(self.user_mapping_file, sep='\t', header=None)
            item_mapping = pd.read_csv(self.item_mapping_file, sep='\t', header=None)

            user_factors_full = self._remap_factors(user_mapping, users_factors, num_users, nf)
            item_factors_full = self._remap_factors(item_mapping, items_factors, num_items, nf)

            if items_biases is not None:
                bias_factors_full = np.zeros(num_items,)
                np.put(bias_factors_full, item_mapping.loc[:, 1].values, items_biases)
                items_biases = bias_factors_full

            _users_factors = user_factors_full
            _items_factors = item_factors_full
        else:
            _users_factors = users_factors['col3'].values.reshape(nu, nf)
            _items_factors = items_factors['col3'].values.reshape(ni, nf)

        self.factors[self.data.fields.userid] = _users_factors
        self.factors[self.data.fields.itemid] = _items_factors
        self._items_biases = items_biases

    def _make_factors_orthogonal(self):
        u = self.factors[self.data.fields.userid]
        v = self.factors[self.data.fields.itemid]

        if self._items_biases is not None:
            b = self._items_biases
            u = np.hstack((u, np.ones((u.shape[0], 1))))
            v = np.hstack((v, b[:, np.newaxis]))

        U, V = self.orthogonalize(u, v)
        self.factors[self.data.fields.userid] = U
        self.factors[self.data.fields.itemid] = V

    def build(self):
        self._recommendations = None
        self._save_to_disk()  # offload train and test sets to disk
        run_code = self._run_external()  # train the model
        if run_code != 0:
            raise ValueError('External program failed.')

        # get saved model parameters
        self._parse_factors()

        # TODO condition on diagonal elements to other elements ratio
        if self.orthogonal_factors or self.data.warm_start:
            self._make_factors_orthogonal()

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        if self.data.warm_start or self.orthogonal_factors:
            return super(MyMediaLiteWrapper, self).slice_recommendations(test_data, shape, start, stop, test_users)

        slice_data = self._slice_test_data(test_data, start, stop)
        user_factors = self.factors[self.data.fields.userid][test_users[start:stop], :]
        item_factors = self.factors[self.data.fields.itemid]
        scores = user_factors.dot(item_factors.T)
        return scores, slice_data
