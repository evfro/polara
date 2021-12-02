import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
try:
    import networkx as nx
except ImportError:
    nx = None
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


def leave_one_out(
        data,
        key = 'userid',
        target = None,
        sample_top = False,
        random_state = None
    ):
    '''
    Samples 1 item per every user according to the rule `sample_top`.
    It always shuffles the input data. The reason is that even if sampling
    top-rated elements, there could be several items with the same top rating.
    '''
    if sample_top: # sample item with the highest target value (e.g., rating, time, etc.)
        idx = (
            data[target]
            .sample(frac=1, random_state=random_state) # handle same feedback for different items
            .groupby(data[key], sort=False)
            .idxmax()
        ).values
    else: # sample random item
        idx = (
            data[key]
            .sample(frac=1, random_state=random_state)
            .drop_duplicates(keep='first') # data is shuffled - simply take the 1st element
            .index
        ).values

    observed = data.drop(idx)
    holdout = data.loc[idx]
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


def earliest_last_out(data, userid='userid', priority='timestamp', copy=False):
    '''
    It helps avoiding "recommendations from future", when training set contains
    events that occur later than some events in the holdout and can therefore
    provide an oracle hint for the algorithm. 
    '''
    holdout_idx, observed_idx, future_idx = split_top_continuous(
        to_numeric_array(data[userid]), data[priority].values
    )
    
    observed = data.iloc[observed_idx]
    holdout = data.iloc[holdout_idx]
    future = data.iloc[future_idx]

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


def pcore_filter(data, pcore, userid, itemid, keep_columns=True):
    if nx is None:
        raise NotImplementedError('pcore filtering requires the `networkx` library installed.')
    g, node_prefix = bipartite_graph_from_df(data, userid, itemid)
    g_pcore = nx.k_core(g, k=pcore) # apply p-core filtering
    pcore_data = pd.DataFrame.from_records(
        read_bipartite_edges(g_pcore, part=0), # iterate user-wise
        columns=[userid, itemid, 'index']
    ).set_index('index')
    # remove user/item node identifiers and restore source dtypes
    for field, prefix in node_prefix.items():
        start = len(prefix)
        pcore_data.loc[:, field] = pcore_data[field].str[start:].astype(data.dtypes[field])
    if keep_columns and (data.shape[1] > 2):
        remaining_data = data.loc[pcore_data.index, data.columns.drop([userid, itemid])]
        pcore_data = pd.concat([pcore_data, remaining_data], axis=1)
    return pcore_data

def bipartite_graph_from_df(df, top, bottom):
    '''
    Construct bipartite top-bottom graph from pandas DataFrame.
    Assumes DataFrame has `top` and `bottom` columns.
    Edge weights are used to store source DataFrame index.
    '''
    node_prefix = {top: 't-', bottom: 'b-'}
    nx_data = (
        df[[top, bottom]]
        .agg({ # add node identifiers for bipartite graph
            top: lambda x: f"{node_prefix[top]}{x}",
            bottom: lambda x: f"{node_prefix[bottom]}{x}",
        })
    )
    g = nx.Graph()
    g.add_nodes_from(nx_data[top].unique(), bipartite=0)
    g.add_nodes_from(nx_data[bottom].unique(), bipartite=1)
    edge_iter = (
        nx_data
        .reset_index()
        [[top, bottom, 'index']] # keep source index
        .itertuples(index=False, name=None)
    )
    g.add_weighted_edges_from(edge_iter)
    return g, node_prefix

def read_bipartite_edges(graph, part=0):
    weighted = nx.is_weighted(graph)
    nodes = (node for node, prop in graph.nodes.items() if prop["bipartite"]==part)
    for node in nodes:
        yield from graph.edges(node, data='weight' if weighted else False)


