import matplotlib.pyplot as plt


def show_hits(all_scores, errors=None, err_alpha = 0.2, figsize=(16, 5), ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    all_scores['hits']['true_positive'].plot(ax=ax[0], legend=False, title='True Positive Hits')
    all_scores['hits']['false_positive'].plot(ax=ax[1], legend=False, title='False Positive Hits')

    if errors:
        errx = errors['hits']['false_positive']
        erry = errors['hits']['true_positive']
        for method in errL.columns:
            x = all_scores['hits']['false_positive'][method]
            y = all_scores['hits']['true_positive'][method]
            lower_boundx = x - errx
            upper_boundx = x + errx
            lower_boundy = y - erry
            upper_boundy = y + erry

            ax.fill_between(x, lower_boundy, upper_boundy, alpha=err_alpha, label='std err')
            ax.fill_betweenx(y, lower_boundx, upper_boundx, alpha=err_alpha, label='std err')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


def show_hit_rates(all_scores, errors=None, err_alpha = 0.2, ROC_middle=True, figsize=(16, 5), limit=True, ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
        show_legend = True
    else:
        show_legend = False
    max_val = 0
    for name in all_scores['relevance'].columns.levels[1]:
        scores = all_scores['relevance'].xs(name, 1, 1).sort_values('fallout')
        new_val = max(scores.recall.max(), scores.fallout.max())
        if (new_val > max_val) and name != 'mostpopular':
            max_val = new_val
        scores.plot.line(x='fallout', y='recall', label=name, ax=ax, legend=False)
        if errors:
            err = errors['relevance'].xs(name, 1, 1).sort_values('fallout')
            x = scores.fallout
            y = scores.recall
            lower_bound = y - err.recall
            upper_bound = y + err.recall
            ax.fill_between(x, lower_bound, upper_bound, alpha=err_alpha, label='std err')
    
    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    if limit:
        ax.set_xlim(0, max_val+0.01)
        ax.set_ylim(0, max_val+0.01)

    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")

    if ROC_middle:
        ax.plot([0, max_val], [0, max_val], linestyle='--', c='grey')



def show_ranking(all_scores, errors=None, err_alpha = 0.2, figsize=(16, 10), limit=False, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        show_legend = True
    else:
        show_legend = False

    all_scores['ranking']['nDCG'].plot(ax=ax[0], legend=False)
    all_scores['ranking']['nDCL'].plot(ax=ax[1], legend=False)
    
    if show_legend:
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            
    if errors:
        errG = errors['ranking']['nDCG']
        errL = errors['ranking']['nDCL']
        for method in errL.columns:
            x = errG.index
            err1 = errG[method]
            err2 = errL[method]
            y1 = all_scores['ranking']['nDCG'][method]
            y2 = all_scores['ranking']['nDCL'][method]
            lower_bound1 = y1 - err1
            upper_bound1 = y1 + err1
            lower_bound2 = y2 - err2
            upper_bound2 = y2 + err2

            ax[0].fill_between(x, lower_bound1, upper_bound1, alpha=err_alpha, label='std err')
            ax[1].fill_between(x, lower_bound2, upper_bound2, alpha=err_alpha, label='std err')

    ax[0].set_ylabel('nDCG@$N$')
    ax[1].set_ylabel('nDCL@$N$')
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))



def show_ranking_positivity(all_scores, ROC_middle=True, figsize=(16, 5), limit=True, ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca()

    max_val = 0
    methods = all_scores['ranking'].columns.levels[1]

    for method in methods:
        scores = all_scores['ranking'].xs(method, 1, 1).sort_values('nDCL')
        new_val = max(scores.nDCG.max(), scores.nDCL.max())
        if (new_val > max_val) and method != 'mostpopular':
            max_val = new_val
        scores.plot.line(x='nDCL', y='nDCG', label=method, ax=ax, legend=False)

    if limit:
        ax.set_xlim(0, max_val+0.01)
        ax.set_ylim(0, max_val+0.01)

    if ROC_middle:
        ax.plot([0, max_val], [0, max_val], linestyle='--', c='grey')

    ax.set_ylabel("Positive Ranking")
    ax.set_xlabel("Negative Ranking")


def show_precision_recall(all_scores, errors=None, err_alpha = 0.2, figsize=(16, 5), limit=False, ax=None):
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        show_legend = True
    else:
        show_legend = False
    
    for name in all_scores['relevance'].columns.levels[1]:
        plot_data = all_scores['relevance'].xs(name, 1, 1).sort_values('recall')
        plot_data.plot.line(x='recall', y='precision', label=name, ax=ax , legend=False)


        if errors:
            #errx = errors['relevance'].xs(name, 1, 1).sort_values('recall').recall
            erry = errors['relevance'].xs(name, 1, 1).sort_values('recall').precision
            x = all_scores['relevance'].xs(name, 1, 1).sort_values('recall').recall
            y = all_scores['relevance'].xs(name, 1, 1).sort_values('recall').precision
            #lower_boundx = x - errx
            #upper_boundx = x + errx
            lower_boundy = y - erry
            upper_boundy = y + erry
            ax.fill_between(x, lower_boundy, upper_boundy, alpha=err_alpha, label='std err')
            #ax.fill_betweenx(y, lower_boundx, upper_boundx, alpha=err_alpha, label='std err')

    if show_legend:
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    if limit:
        lim = max(all_scores['relevance']['precision'].max().max(),
                    all_scores['relevance']['recall'].max().max()) + 0.05
        plt.xlim((0, lim));
        plt.ylim((0, lim));


def show_relevance(all_scores, figsize=(16, 10), ax=None):
    if not ax:
        fig, ax = plt.subplots(2, 2, figsize=figsize)

    all_scores['relevance']['precision'].plot(ax=ax[0, 0], legend=False, title='Precision@$N$')
    all_scores['relevance']['recall'].plot(ax=ax[0, 1], title='Recall@$N$', legend=False)
    all_scores['relevance']['fallout'].plot(ax=ax[1, 0], title='Fallout@$N$', legend=False)
    all_scores['relevance']['miss_rate'].plot(ax=ax[1, 1], title='Miss Rate@$N$', legend=False)
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
