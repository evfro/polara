# python 2/3 interoperability
from __future__ import print_function
try:
    range = xrange
except NameError:
    pass

from math import sqrt
import pandas as pd


def sample_ci(df, coef=2.776, level=None): # 95% CI for sample under Student's t-test
    # http://www.stat.yale.edu/Courses/1997-98/101/confint.htm
    # example from http://onlinestatbook.com/2/estimation/mean.html
    if isinstance(level, str):
        level = df.index.names.index(level)

    nlevels = df.index.nlevels
    if (nlevels == 1) & (level is None):
        n = df.shape[0]
    elif (nlevels==2) & (level is not None):
        n = df.index.levshape[1-level]
    else:
        raise ValueError
    return coef * df.std(level=level, ddof=1) / sqrt(n)


def save_scores(scores, dataset_name, experiment_name, save_folder=None):
    experiment_keys = scores.keys()
    save_folder = save_folder or 'results'
    path = '{saveto}/{{}}_{{}}_({{}})_{{}}.csv'.format(saveto=save_folder)

    for key in experiment_keys:
        metrics = scores[key].keys()
        for metric in metrics:
            scores[key][metric].to_csv(path.format(dataset_name, experiment_name, key, metric))


def average_results(scores):
    averaged = {}
    errors = {}
    metrics = scores.keys()
    for metric in metrics:
        values = scores[metric].mean(level=1, axis=0).sort_index(axis=1)
        std_err = scores[metric].std(level=1, axis=0).sort_index(axis=1)
        averaged[metric] = values
        errors[metric] = std_err

    return averaged, errors


def evaluate_models(models, metrics, **kwargs):
    scores = []
    for model in models:
        model_scores = model.evaluate(metric_type=metrics, **kwargs)
        # ensure correct format
        model_scores = model_scores if isinstance(model_scores, list) else [model_scores]
        # concatenate all scores
        name = [model.method]
        metric_types = [s.__class__.__name__.lower() for s in model_scores]
        scores_df = pd.concat([pd.DataFrame([s], index=name) for s in model_scores],
                              keys=metric_types, axis=1)
        scores.append(scores_df)
    res = pd.concat(scores, axis=0)
    res.columns.names = ['type', 'metric']
    res.index.names = ['model']
    return res


def set_topk(models, topk):
    for model in models:
        model.topk = topk


def build_models(models, force=True):
    for model in models:
        if not model._is_ready or force:
            model.build()


def consolidate(scores, level_name=None, level_keys=None):
    level_names = [level_name] + scores[0].index.names
    return pd.concat(scores, axis=0, keys=level_keys, names=level_names)


def holdout_test(models, holdout_sizes=[1], metrics='all'):
    holdout_scores = []
    data = models[0].data
    assert all([model.data is data for model in models[1:]]) #check that data is shared across models

    for i in holdout_sizes:
        data.holdout_size = i
        data.update()
        metric_scores = evaluate_models(models, metrics)
        holdout_scores.append(metric_scores)
    return consolidate(holdout_scores, level_name='hsize', level_keys=holdout_sizes)


def topk_test(models, **kwargs):
    metrics = kwargs.pop('metrics', None) or 'all'
    topk_list = kwargs.pop('topk_list', [10])
    topk_scores = []
    data = models[0].data
    assert all([model.data is data for model in models[1:]]) # check that data is shared across models

    topk_list_sorted = list(reversed(sorted(topk_list))) # start from max topk and rollback

    for topk in topk_list_sorted:
        kwargs['topk'] = topk
        metric_scores = evaluate_models(models, metrics, **kwargs)
        topk_scores.append(metric_scores)

    level_name = 'top-n'
    res = consolidate(topk_scores, level_name=level_name, level_keys=topk_list_sorted)
    return res.sort_index(level=level_name, sort_remaining=False)


def run_cv_experiment(models, folds=None, metrics='all', fold_experiment=evaluate_models,
                      force_build=True, deferred_update=False, iterator=lambda x: x, **kwargs):
    if not isinstance(models, (list, tuple)):
        models = [models]

    data = models[0].data
    assert all([model.data is data for model in models[1:]]) # check that data is shared across models

    if folds is None:
        folds = range(1, int(1/data.test_ratio) + 1)

    fold_results = []
    for fold in iterator(folds):
        data.test_fold = fold
        build_models(models, force_build)
        if not deferred_update:
            data.update() # data configuration is assumed to be fixed during CV
        fold_result = fold_experiment(models, metrics=metrics, **kwargs)
        fold_results.append(fold_result)
    return consolidate(fold_results, level_name='fold', level_keys=folds)
