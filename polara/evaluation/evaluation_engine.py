import pandas as pd


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


def evaluate_models(models, metrics, topk=None):
    metric_scores = []
    for metric in metrics:
        model_scores = []
        for model in models:
            #print 'model {}'.format(model.method)
            scores = model.evaluate(method=metric, topk=topk)
            model_scores.append(scores)
        metric_scores.append(pd.DataFrame(model_scores, index=[model.method for model in models]).T)
    return metric_scores


def set_topk(models, topk):
    for model in models:
        model.topk = topk


def build_models(models):
    for model in models:
        model.build()


def refresh_models(models):
    for model in models:
        model._recommendations = None


def consolidate(scores, params, metrics):
    res = {}
    for i, metric in enumerate(metrics):
        res[metric] = pd.concat([scores[j][i] for j in range(len(params))],
                                keys=params).unstack().swaplevel(0, 1, 1).sort_index()
    return res


def consolidate_folds(scores, folds, metrics, index_names = ['fold', 'top-n']):
    res = {}
    for metric in metrics:
        data = pd.concat([scores[j][metric] for j in folds], keys=folds)
        data.index.names = index_names
        res[metric] = data
    return res


def holdout_test_pair(model1, model2, holdout_sizes=[1], metrics=['hits']):
    holdout_scores = []
    models = [model1, model2]
    check_updates(models)

    data1 = model1.data
    data2 = model2.data
    for i in holdout_sizes:
        print i,
        data1.holdout_size = i
        data2.holdout_size = i
        data1.update()
        data1.update()

        refresh_models(models)
        metric_scores = evaluate_models(models, metrics)
        holdout_scores.append(metric_scores)

    return consolidate(holdout_scores, holdout_sizes, metrics)


def check_updates(models):
    data = models[0].data
    if data.has_changed: #Rebuild models entirely
        print 'Data has been changed. Rebuiding the models.'
        build_models(models)
    elif data.has_updated: #just force recommendations renewal
        print 'Test data has been updated. Refreshing the models.'
        refresh_models(models)


def holdout_test(models, holdout_sizes=[1], metrics=['hits']):
    #check_updates(models) #will rebuild or renew models if data was manipulated in previous experiments
    holdout_scores = []
    data = models[0].data
    assert all([model.data is data for model in models[1:]]) #check that data is shared across models

    for i in holdout_sizes:
        print i,
        data.holdout_size = i
        data.update() #can be omitted but it's more safe
        refresh_models(models) #test data is updated - clear old recommendations

        metric_scores = evaluate_models(models, metrics)
        holdout_scores.append(metric_scores)

    return consolidate(holdout_scores, holdout_sizes, metrics)


def topk_test(models, topk_list=[10], metrics=['hits']):
    topk_scores = []
    data = models[0].data
    data.update()
    check_updates(models) #will rebuild or renew models if data was manipulated in previous experiments
    assert all([model.data is data for model in models[1:]]) #check that data is shared across models
    topk_list = list(reversed(sorted(topk_list))) #start from max topk and rollback

    for topk in topk_list:
        print topk,
        metric_scores = evaluate_models(models, metrics, topk)
        topk_scores.append(metric_scores)

    return consolidate(topk_scores, topk_list, metrics)
