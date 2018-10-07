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


def consolidate(scores, params, metrics):
    res = {}
    for i, metric in enumerate(metrics):
        res[metric] = pd.concat([scores[j][i] for j in range(len(params))],
                                keys=params).unstack().swaplevel(0, 1, 1).sort_index()
    return res


def consolidate_folds(scores, folds, metrics, index_names=['fold', 'top-n']):
    res = {}
    for metric in metrics:
        data = pd.concat([scores[j][metric] for j in folds], keys=folds)
        data.index.names = index_names
        res[metric] = data
    return res


def holdout_test_pair(model1, model2, holdout_sizes=[1], metrics=['hits']):
    holdout_scores = []
    models = [model1, model2]

    data1 = model1.data
    data2 = model2.data
    for i in holdout_sizes:
        print(i, end=' ')
        data1.holdout_size = i
        data1.update()
        data2.holdout_size = i
        data2.update()

        metric_scores = evaluate_models(models, metrics)
        holdout_scores.append(metric_scores)

    return consolidate(holdout_scores, holdout_sizes, metrics)


def holdout_test(models, holdout_sizes=[1], metrics=['hits'], force_build=True):
    holdout_scores = []
    data = models[0].data
    assert all([model.data is data for model in models[1:]]) #check that data is shared across models

    build_models(models, force_build)
    for i in holdout_sizes:
        data.holdout_size = i
        data.update()

        metric_scores = evaluate_models(models, metrics)
        holdout_scores.append(metric_scores)

    return consolidate(holdout_scores, holdout_sizes, metrics)


def topk_test(models, topk_list=[10], metrics=['hits'], force_build=True):
    topk_scores = []
    data = models[0].data
    assert all([model.data is data for model in models[1:]]) #check that data is shared across models

    data.update()
    topk_list = list(reversed(sorted(topk_list))) #start from max topk and rollback

    build_models(models, force_build)
    for topk in topk_list:
        metric_scores = evaluate_models(models, metrics, topk)
        topk_scores.append(metric_scores)

    return consolidate(topk_scores, topk_list, metrics)
