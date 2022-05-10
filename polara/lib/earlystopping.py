from polara.tools.display import log_status


def early_stopping_callback(evaluate_func, margin=0, max_fails=1, verbose=True, logger=None):
    assert margin >= 0
    assert max_fails >= 0
    if logger is None:
        logger = log_status
    def check_metric_growth(step, *args, **kwargs):
        # validate model
        current_score = evaluate_func(*args, **kwargs)
        logger(f'Step {step} metric score: {current_score}', verbose=verbose)
        # track progress
        previous_best = check_metric_growth.target
        check_metric_growth.history.append(current_score)
        if current_score > previous_best:
            check_metric_growth.target = current_score
            check_metric_growth.iter = step
        # early stopping
        try:
            check_early_stop(current_score, previous_best, margin=margin, max_fails=max_fails)
        except StopIteration as e:
            best_res = check_metric_growth.target
            num_iter = check_metric_growth.iter + 1
            logger(
                f'Metric no longer improves. Best score {best_res}, attained in {num_iter} iterations.',
                verbose=verbose
            )
            raise e
    check_metric_growth.target = float('-inf')
    check_metric_growth.iter = None
    check_metric_growth.history = []
    return check_metric_growth


def check_early_stop(target_score, previous_best, margin, max_fails):
    if target_score <= previous_best + margin:
        check_early_stop.fail_count += 1
    else:
        check_early_stop.fail_count = 0
    if check_early_stop.fail_count >= max_fails:
        raise StopIteration


def evaluator(model, target_metric):
    def iter_evaluate(*args, **kwargs):
        model.reuse_model(*args, **kwargs)
        metrics = model.evaluate()
        return find_target_metric(metrics, target_metric)
    return iter_evaluate


def find_target_metric(metrics, target_metric):
    'Convenience function to quickly extract the required metric.'
    for metric in metrics:
        if hasattr(metric, target_metric):
            return getattr(metric, target_metric)
    raise ValueError('Target metric not found.')