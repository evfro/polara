import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import is_numeric_dtype
from polara.lib.sampler import split_top_continuous
from polara.tools.random import check_random_state


def reindex(raw_data, index, filter_invalid=True, names=None):
    '''
    Factorizes column values based on provided pandas index. Allows resetting
    index names. Optionally drops rows with entries not present in the index.
    '''
    if isinstance(index, pd.Index):
        index = [index]

    if isinstance(names, str):
        names = [names]

    if isinstance(names, (list, tuple, pd.Index)):
        for i, name in enumerate(names):
            index[i].name = name

    new_data = raw_data.assign(**{
        idx.name: idx.get_indexer(raw_data[idx.name]) for idx in index
    })

    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            ' or '.join([f'{idx.name} == -1' for idx in index])
        )
        if maybe_invalid.any():
            print(f'Filtered {maybe_invalid.sum()} invalid observations.')
            new_data = new_data.loc[~maybe_invalid]

    return new_data


def matrix_from_observations(
        data,
        userid='userid',
        itemid='itemid',
        user_index=None,
        item_index=None,
        feedback=None,
        preserve_order=False,
        shape=None,
        dtype=None
    ):
    '''
    Encodes pandas dataframe into sparse matrix. If index is not provided,
    returns new index mapping, which optionally preserves order of original data.
    Automatically removes incosnistent data not present in the provided index.
    '''
    if (user_index is None) or (item_index is None):
        useridx, user_index = pd.factorize(data[userid], sort=preserve_order)
        itemidx, item_index = pd.factorize(data[itemid], sort=preserve_order)
        user_index.name = userid
        item_index.name = itemid
    else:
        data = reindex(data, (user_index, item_index), filter_invalid=True)
        useridx = data[userid].values
        itemidx = data[itemid].values
        if shape is None:
            shape = (len(user_index), len(item_index))

    if feedback is None:
        values = np.ones_like(itemidx, dtype=dtype)
    else:
        values = data[feedback].values

    matrix = csr_matrix((values, (useridx, itemidx)), dtype=dtype, shape=shape)
    return matrix, user_index, item_index


def split_holdout(
        data,
        userid = 'userid',
        feedback = None,
        sample_max_rated = False,
        random_state = None
    ):
    '''
    Samples 1 item per every user according to the rule sample_max_rated.
    It always shuffles the input data. The reason is that even if sampling
    top-rated elements, there could be several items with the same top rating.
    '''
    idx_grouper = (
        data
        .sample(frac=1, random_state=random_state) # randomly permute data
        .groupby(userid, as_index=False, sort=False)
    )
    if sample_max_rated: # take single item with the highest score
        idx = idx_grouper[feedback].idxmax()
    else: # data is already shuffled - simply take the 1st element
        idx = idx_grouper.head(1).index # sample random element

    observed = data.drop(idx.values)
    holdout = data.loc[idx.values]
    return observed, holdout


def sample_unseen_items(item_group, item_pool, n, random_state):
    'Helper function to run on pandas dataframe grouper'
    seen_items = item_group.values
    candidates = np.setdiff1d(item_pool, seen_items, assume_unique=True)
    return random_state.choice(candidates, n, replace=False)


def sample_unseen_interactions(
        data,
        item_pool,
        n_random = 999,
        random_state = None,
        userid = 'userid',
        itemid = 'itemid'
    ):
    '''
    Randomized sampling of unseen items per every user in data. Assumes data
    was already preprocessed to contiguous index.
    '''
    random_state = check_random_state(random_state)
    return (
        data
        .groupby(userid, sort=False)[itemid]
        .apply(sample_unseen_items, item_pool, n_random, random_state)
    )


def verify_split(train, test, random_holdout, feedback, userid='userid'):
    if random_holdout:
        return
    hold_gr = test.set_index(userid)[feedback]
    useridx = hold_gr.index
    train_gr = train.query(f'{userid} in @useridx').groupby(userid)[feedback]
    assert train_gr.apply(lambda x: x.le(hold_gr.loc[x.name]).all()).all()


def to_numeric_array(series):
    if not is_numeric_dtype(series):
        if not hasattr(series, 'cat'):
            series = series.astype('category')
        return series.cat.codes.values
    return series.values


def split_earliest_last(data, userid='userid', priority='timestamp', copy=False):
    '''
    It helps avoiding "recommendations from future", when training set contains events that occur later than some events in the holdout and can therefore provide an oracle hint for the algorithm. 
    '''
    topseq_idx, lowseq_idx, nonseq_idx = split_top_continuous(
        to_numeric_array(data[userid]), data[priority].values
    )
    
    observed = data.iloc[lowseq_idx]
    holdout = data.iloc[topseq_idx]
    future = data.iloc[nonseq_idx]

    if copy:
        observed = observed.copy()
        holdout = holdout.copy()
        future = future.copy()
    
    return observed, holdout, future


def filter_sessions_by_length(data, session_label='userid', min_session_length=3):
    """Filters users with insufficient number of items"""
    if data.duplicated().any():
        raise NotImplementedError

    sz = data[session_label].value_counts(sort=False)
    has_valid_session_length = sz >= min_session_length
    if not has_valid_session_length.all():
        valid_sessions = sz.index[has_valid_session_length]
        new_data = data[data[session_label].isin(valid_sessions)].copy()
        print('Sessions are filtered by length')
    else:
        new_data = data
    return new_data