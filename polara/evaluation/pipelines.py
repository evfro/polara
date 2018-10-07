from __future__ import print_function

from operator import mul as mul_op
from functools import reduce
from random import choice
import pandas as pd


def random_chooser():
    while True:
        values = yield
        yield choice(values)


def random_grid(params, n=60, grid_cache=None):
    if not isinstance(n, int):
        raise TypeError('n must be an integer, not {}'.format(type(n)))
    if n < 0:
        raise ValueError('n should be >= 0')

    grid = grid_cache or set()
    max_n = reduce(mul_op, [len(vals) for vals in params.values()])
    n = min(n if n > 0 else max_n, max_n)
    param_chooser = random_chooser()
    try:
        while len(grid) < n:
            level_choice = []
            for v in params.values():
                next(param_chooser)
                level_choice.append(param_chooser.send(v))
            grid.add(tuple(level_choice))
    except KeyboardInterrupt:
        print('Interrupted by user. Providing current results.')
    return grid


def set_config(model, attributes, values):
    for name, value in zip(attributes, values):
        setattr(model, name, value)


def evaluate_models(models, target_metric='precision', metric_type='all', **kwargs):
    if not isinstance(models, (list, tuple)):
        models = [models]

    model_scores = {}
    for model in models:
        scores = model.evaluate(metric_type, **kwargs)
        scores = [scores] if not isinstance(scores, list) else scores
        scores_df = pd.concat([pd.DataFrame([s]) for s in scores], axis=1)
        if isinstance(target_metric, str):
            model_scores[model.method] = scores_df[target_metric].squeeze()
        elif callable(target_metric):
            model_scores[model.method] = scores_df.apply(target_metric, axis=1).squeeze()
        else:
            raise NotImplementedError
    return model_scores


def find_optimal_svd_rank(model, ranks, target_metric, return_scores=False,
                          protect_factors=True, config=None, verbose=False,
                          ranger=lambda x: x, **kwargs):
    model_verbose = model.verbose
    if config:
        set_config(model, *zip(*config.items()))

    model.rank = svd_rank = max(max(ranks), model.rank)
    if not model._is_ready:
        model.verbose = verbose
        model.build()

    if protect_factors:
        svd_factors = dict(**model.factors) # avoid accidental overwrites

    res = {}
    try:
        for rank in ranger(list(reversed(sorted(ranks)))):
            model.rank = rank
            res[rank] = evaluate_models(model, target_metric, **kwargs)[model.method]
    finally:
        if protect_factors:
            model._rank = svd_rank
            model.factors = svd_factors
        model.verbose = model_verbose

    scores = pd.Series(res)
    best_rank = scores.idxmax()
    if return_scores:
        scores.index.name = 'rank'
        return best_rank, scores.loc[ranks]
    return best_rank


def find_optimal_tucker_ranks(model, tucker_ranks, target_metric, return_scores=False,
                              config=None, verbose=False, same_space=False,
                              ranger=lambda x: x, **kwargs):
    model_verbose = model.verbose
    if config:
        set_config(model, *zip(*config.items()))

    model.mlrank = tuple([max(mode_ranks) for mode_ranks in tucker_ranks])

    if not model._is_ready:
        model.verbose = verbose
        model.build()

    factors = dict(**model.factors)
    tucker_rank = model.mlrank

    res_score = {}
    for r1 in ranger(tucker_ranks[0]):
        for r2 in tucker_ranks[1]:
            if same_space and (r2 != r1):
                continue
            for r3 in tucker_ranks[2]:
                if (r1*r2 < r3) or (r1*r3 < r2) or (r2*r3 < r1):
                    continue
                try:
                    model.mlrank = mlrank = (r1, r2, r3)
                    res_score[mlrank] = evaluate_models(model, target_metric, **kwargs)[model.method]
                finally:
                    model._mlrank = tucker_rank
                    model.factors = dict(**factors)
    model.verbose = model_verbose

    scores = pd.Series(res_score).sort_index()
    best_mlrank = scores.idxmax()
    if return_scores:
        scores.index.names = ['r1', 'r2', 'r3']
        return best_mlrank, scores
    return best_mlrank


def find_optimal_config(model, param_grid, param_names, target_metric, return_scores=False,
                        config=None, reset_config=None, verbose=False, force_build=True,
                        ranger=lambda x: x, **kwargs):
    model_verbose = model.verbose
    if config:
        set_config(model, *zip(*config.items()))

    model.verbose = verbose
    grid_results = {}
    for params in ranger(param_grid):
        set_config(model, param_names, params)

        if not model._is_ready or force_build:
            model.build()
        grid_results[params] = evaluate_models(model, target_metric, **kwargs)[model.method]

        if isinstance(reset_config, dict):
            set_config(model, *zip(*reset_config.items()))
        elif callable(reset_config):
            reset_config(model)
        else:
            raise NotImplementedError

    model.verbose = model_verbose
    # workaround non-orderable configs (otherwise pandas raises error)
    scores = pd.Series(**dict(zip(('index', 'data'),
                                  (zip(*grid_results.items())))))
    best_config = scores.idxmax()
    if return_scores:
        scores.index.names = param_names
        return best_config, scores
    return best_config
