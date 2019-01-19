import numpy as np
import scipy as sp
import pandas as pd


def compute_graph_laplacian(edges, index):
    all_edges = set()
    for a, b in edges:
        try:
            a = index.get_loc(a)
            b = index.get_loc(b)
        except KeyError:
            continue
        if a == b: # exclude self links
            continue
        # make graph undirectional
        all_edges.add((a, b))
        all_edges.add((b, a))

    sp_edges = sp.sparse.csr_matrix((np.ones(len(all_edges)), zip(*all_edges)))
    assert (sp_edges.diagonal() == 0).all()
    return sp.sparse.csgraph.laplacian(sp_edges).tocsr(), sp_edges


def get_epinions_data(ratings_path=None, trust_data_path=None):
    res = []
    if ratings_path:
        ratings = pd.read_csv(ratings_path,
                              delim_whitespace=True,
                              skiprows=[0],
                              skipfooter=1,
                              engine='python',
                              header=None,
                              skipinitialspace=True,
                              names=['user', 'film', 'rating'],
                              usecols=['user', 'film', 'rating'])
        res.append(ratings)

    if trust_data_path:
        edges = pd.read_table(trust_data_path,
                              delim_whitespace=True,
                              skiprows=[0],
                              skipfooter=1,
                              engine='python',
                              header=None,
                              skipinitialspace=True,
                              usecols=[0, 1])
        res.append(edges)

    if len(res)==1: res = res[0]
    return res
