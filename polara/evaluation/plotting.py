import matplotlib.pyplot as plt


def _plot_pair(scores, keys, titles=None, errors=None, err_alpha = 0.2, figsize=(16, 5), ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        show_legend = True
    else:
        show_legend = False

    left, right = keys
    left_title, right_title = titles or keys

    scores[left].plot(ax=ax[0], legend=False)
    scores[right].plot(ax=ax[1], legend=False)

    if show_legend:
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    if errors is not None:
        errG = errors[left]
        errL = errors[right]
        for method in errL.columns:
            x = errG.index
            err1 = errG[method]
            err2 = errL[method]
            y1 = scores[left][method]
            y2 = scores[right][method]
            lower_bound1 = y1 - err1
            upper_bound1 = y1 + err1
            lower_bound2 = y2 - err2
            upper_bound2 = y2 + err2

            ax[0].fill_between(x, lower_bound1, upper_bound1, alpha=err_alpha, label='std err')
            ax[1].fill_between(x, lower_bound2, upper_bound2, alpha=err_alpha, label='std err')

    ax[0].set_ylabel(left_title)
    ax[1].set_ylabel(right_title)


def show_hits(all_scores, **kwargs):
    scores = all_scores['hits']
    keys = ['true_positive', 'false_positive']
    kwargs['titles'] = ['True Positive Hits @$n$', 'False Positive Hits @$n$']
    kwargs['errors'] = kwargs['errors']['hits'] if kwargs.get('errors', False) else None
    _plot_pair(scores, keys, **kwargs)


def show_ranking(all_scores, **kwargs):
    scores = all_scores['ranking']
    keys = ['nDCG', 'nDCL']
    kwargs['titles'] = ['nDCG@$n$', 'nDCL@$n$']
    kwargs['errors'] = kwargs['errors']['ranking'] if kwargs.get('errors', False) else None
    _plot_pair(scores, keys, **kwargs)


def _cross_plot(scores, keys, titles=None, errors=None, err_alpha = 0.2, ROC_middle=False, figsize=(8, 5), limit=None, ax=None):
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        show_legend = True
    else:
        show_legend = False

    methods = scores.columns.levels[1]
    x, y = keys
    for method in methods:
        plot_data = scores.xs(method, 1, 1).sort_values(x)
        plot_data.plot.line(x=x, y=y, label=method, ax=ax, legend=False)

    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    if errors is not None:
        for method in methods:
            plot_data = scores.xs(method, 1, 1).sort_values(x)
            error = errors.xs(method, 1, 1).sort_values(x)
            x_val = plot_data[x]
            y_val = plot_data[y]
            lower_bound = y_val - error[y]
            upper_bound = y_val + error[y]
            ax.fill_between(x_val, lower_bound, upper_bound, alpha=err_alpha, label='std err')

    if limit:
        if not isinstance(limit, (tuple, list)):
            limit = (0, limit)
        ax.set_xlim(*limit)
        ax.set_ylim(*limit)

    titles = titles or keys
    ax.set_xlabel(titles[0])
    ax.set_ylabel(titles[1])

    if ROC_middle:
        lims = ax.get_xlim()
        ax.plot(lims, lims, linestyle='--', c='grey')


def show_hit_rates(all_scores, **kwargs):
    scores = all_scores['relevance']
    keys = ['fallout', 'recall']
    kwargs['titles'] = ['False Positive Rate', 'True Positive Rate']
    kwargs['errors'] = kwargs['errors']['relevance'] if kwargs.get('errors', False) else None
    kwargs['ROC_middle'] = True
    kwargs['limit'] = max(scores['fallout'].max().max(), scores['recall'].max().max()) + 0.01
    _cross_plot(scores, keys, **kwargs)


def show_ranking_positivity(all_scores, **kwargs):
    scores = all_scores['ranking']
    keys = ['nDCL', 'nDCG']
    kwargs['titles'] = ['Negative Ranking', 'Positive Ranking']
    kwargs['errors'] = kwargs['errors']['ranking'] if kwargs.get('errors', False) else None
    kwargs['ROC_middle'] = True
    kwargs['limit'] = max(scores['nDCL'].max().max(), scores['nDCG'].max().max()) + 0.01
    _cross_plot(scores, keys, **kwargs)


def show_precision_recall(all_scores, limit=False, ignore_field_limit=None, **kwargs):
    scores = all_scores['relevance']
    keys = ['recall', 'precision']
    kwargs['titles'] = ['Recall', 'Precision']
    kwargs['errors'] = kwargs['errors']['relevance'] if kwargs.get('errors', False) else None
    kwargs['ROC_middle'] = False
    if limit:
        maxx = scores['recall'].drop(ignore_field_limit, axis=1, errors='ignore').max().max()
        maxy = scores['precision'].drop(ignore_field_limit, axis=1, errors='ignore').max().max()
        kwargs['limit'] = max(maxx, maxy) + 0.05
    _cross_plot(scores, keys, **kwargs)


def show_relevance(all_scores, figsize=(16, 10), ax=None):
    if not ax:
        fig, ax = plt.subplots(2, 2, figsize=figsize)

    all_scores['relevance']['precision'].plot(ax=ax[0, 0], legend=False, title='Precision@$N$')
    all_scores['relevance']['recall'].plot(ax=ax[0, 1], title='Recall@$N$', legend=False)
    all_scores['relevance']['fallout'].plot(ax=ax[1, 0], title='Fallout@$N$', legend=False)
    all_scores['relevance']['miss_rate'].plot(ax=ax[1, 1], title='Miss Rate@$N$', legend=False)
