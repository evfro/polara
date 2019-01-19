from math import sqrt
import numpy as np
from numba import jit, njit, prange

from polara.tools.timing import track_time


@njit(nogil=True)
def mf_sgd_sweep(users_idx, items_idx, feedbacks, P, Q, eta, lambd, *args):
    cum_error = 0
    for k, a in enumerate(feedbacks):
        i = users_idx[k]
        j = items_idx[k]

        pi = P[i, :]
        qj = Q[j, :]

        e = a - pi @ qj

        new_pi = pi + eta * (e*qj - lambd*pi)
        new_qj = qj + eta * (e*pi - lambd*qj)

        P[i, :] = new_pi
        Q[j, :] = new_qj

        cum_error += e*e
    return cum_error

@njit(nogil=True)
def mf_sgd_sweep_biased(users_idx, items_idx, feedbacks, P, Q, eta, lambd,
                        b_user, b_item, mu, *args):
    cum_error = 0
    for k, a in enumerate(feedbacks):
        i = users_idx[k]
        j = items_idx[k]

        pi = P[i, :]
        qj = Q[j, :]
        bi = b_user[i]
        bj = b_item[j]

        e = a - (pi @ qj + bi + bj + mu)

        new_pi = pi + eta * (e*qj - lambd*pi)
        new_qj = qj + eta * (e*pi - lambd*qj)

        P[i, :] = new_pi
        Q[j, :] = new_qj

        new_bi = bi + eta * (e - lambd*bi)
        new_bj = bj + eta * (e - lambd*bj)

        b_user[i] = new_bi
        b_item[j] = new_bj

        cum_error += e*e
    return cum_error



@njit(nogil=True)
def identity(x, *args): # used to fall back to standard SGD
    return x


@njit(nogil=True)
def adagrad(grad, m, cum_sq_grad, smoothing=1e-6):
    cum_sq_grad_update = cum_sq_grad[m, :] + grad * grad
    cum_sq_grad[m, :] = cum_sq_grad_update
    adjusted_grad = grad / (smoothing + np.sqrt(cum_sq_grad_update))
    return adjusted_grad


@njit(nogil=True)
def rmsprop(grad, m, cum_sq_grad, gamma=0.9, smoothing=1e-6):
    cum_sq_grad_update = gamma * cum_sq_grad[m, :] + (1 - gamma) * (grad * grad)
    cum_sq_grad[m, :] = cum_sq_grad_update
    adjusted_grad = grad / (smoothing + np.sqrt(cum_sq_grad_update))
    return adjusted_grad


@njit(nogil=True)
def adam(grad, m, cum_grad, cum_sq_grad, step, beta1=0.9, beta2=0.999, smoothing=1e-6):
    cum_grad_update = beta1 * cum_grad[m, :] + (1 - beta1) * grad
    cum_grad[m, :] = cum_grad_update
    cum_sq_grad_update = beta2 * cum_sq_grad[m, :] + (1 - beta2) * (grad * grad)
    cum_sq_grad[m, :] = cum_sq_grad_update
    step[m] = t = step[m] + 1
    db1 = 1 - beta1**t
    db2 = 1 - beta2**t
    adjusted_grad = cum_grad_update/db1 / (smoothing + np.sqrt(cum_sq_grad_update/db2))
    return adjusted_grad


@njit(nogil=True)
def adanorm(grad, m, smoothing=1e-6):
    gnorm2 = grad @ grad
    adjusted_grad = grad / sqrt(smoothing + gnorm2)
    return adjusted_grad

@njit(nogil=True)
def gnprop(grad, m, cum_sq_norm, gamma=0.99, smoothing=1e-6):
    cum_sq_norm_update = gamma * cum_sq_norm[m] + (1 - gamma) * (grad @ grad)
    cum_sq_norm[m] = cum_sq_norm_update
    adjusted_grad = grad / sqrt(smoothing + cum_sq_norm_update)
    return adjusted_grad


@njit(nogil=True, parallel=False)
def generalized_sgd_sweep(row_idx, col_idx, values, P, Q,
                          eta, lambd, row_nnz, col_nnz,
                          apply_kernel, kernel_params,
                          adjust_gradient, adjustment_params):
    cum_error = 0
    for k, val in enumerate(values):
        m = row_idx[k]
        n = col_idx[k]

        pm = P[m, :]
        qn = Q[n, :]

        err = val - pm @ qn
        row_lambda = lambd / row_nnz[m]
        col_lambda = lambd / col_nnz[n]

        kpm = apply_kernel(pm, P, m, *kernel_params[0])
        ngrad_p = err * qn - kpm * row_lambda
        sqn = apply_kernel(qn, Q, n, *kernel_params[1])
        ngrad_q = err * pm - sqn * col_lambda

        adjusted_ngrad_p = adjust_gradient(ngrad_p, m, *adjustment_params[0])
        new_pm = pm + eta * adjusted_ngrad_p
        adjusted_ngrad_q = adjust_gradient(ngrad_q, n, *adjustment_params[1])
        new_qn = qn + eta * adjusted_ngrad_q

        P[m, :] = new_pm
        Q[n, :] = new_qn

        cum_error += err*err
    return cum_error


# @jit
def mf_sgd_boilerplate(interactions, shape, nonzero_count, rank,
                       lrate, lambd, num_epochs, tol,
                       sgd_sweep_func=None,
                       apply_kernel=None, kernel_params=None,
                       adjust_gradient=None, adjustment_params=None,
                       seed=None, verbose=False,
                       iter_errors=None, iter_time=None):
    assert isinstance(interactions, tuple) # required by numba
    assert isinstance(nonzero_count, tuple) # required by numba

    nrows, ncols = shape
    row_shp = (nrows, rank)
    col_shp = (ncols, rank)

    rnds = np.random if seed is None else np.random.RandomState(seed)
    row_factors = rnds.normal(scale=0.1, size=row_shp)
    col_factors = rnds.normal(scale=0.1, size=col_shp)

    sgd_sweep_func = sgd_sweep_func or generalized_sgd_sweep
    apply_kernel = apply_kernel or identity
    kernel_params = kernel_params or ((), ())
    adjust_gradient = adjust_gradient or identity
    adjustment_params = adjustment_params or ((), ())

    nnz = len(interactions[-1])
    last_err = np.finfo(np.float64).max
    training_time = []
    for epoch in range(num_epochs):
        if adjust_gradient in [adagrad, rmsprop]:
            adjustment_params = ((np.zeros(row_shp, dtype='f8'),),
                                 (np.zeros(col_shp, dtype='f8'),)
                                )
        if adjust_gradient is gnprop:
            adjustment_params = ((np.zeros(nrows, dtype='f8'),),
                                 (np.zeros(ncols, dtype='f8'),)
                                )
        if adjust_gradient is adam:
            adjustment_params = ((np.zeros(row_shp, dtype='f8'),
                                  np.zeros(row_shp, dtype='f8'),
                                  np.zeros(nrows, dtype='intp')),
                                 (np.zeros(col_shp, dtype='f8'),
                                  np.zeros(col_shp, dtype='f8'),
                                  np.zeros(ncols, dtype='intp'))
                                )

        with track_time(training_time, verbose=False):
            new_err = sgd_sweep_func(*interactions, row_factors, col_factors,
                                     lrate, lambd, *nonzero_count,
                                     apply_kernel, kernel_params,
                                     adjust_gradient, adjustment_params)

        refined = abs(last_err - new_err) / last_err
        last_err = new_err
        rmse = sqrt(new_err / nnz)
        if iter_errors is not None:
            iter_errors.append(rmse)
        if verbose:
            print('Epoch: {}. RMSE: {}'.format(epoch, rmse))
        if refined < tol:
            break
    if iter_time is not None:
        iter_time.extend(training_time)
    return row_factors, col_factors


def simple_mf_sgd(interactions, shape, nonzero_count, rank, lrate, lambd, num_epochs, tol,
                  seed=None, verbose=False, iter_errors=None):
    nonzero_count = ()
    return mf_sgd_boilerplate(interactions, shape, nonzero_count, rank,
                              lrate, lambd, num_epochs, tol,
                              sgd_sweep_func=mf_sgd_sweep,
                              seed=seed, verbose=verbose, iter_errors=iter_errors)


def simple_pmf_sgd(interactions, shape, nonzero_count, rank,
                   lrate, sigma, num_epochs, tol,
                   adjust_gradient=None, adjustment_params=None,
                   seed=None, verbose=False,
                   iter_errors=None, iter_time=None):
    lambd = 0.5 * sigma**2
    return mf_sgd_boilerplate(interactions, shape, nonzero_count, rank,
                              lrate, lambd, num_epochs, tol,
                              adjust_gradient=adjust_gradient,
                              adjustment_params=adjustment_params,
                              seed=seed, verbose=verbose,
                              iter_errors=iter_errors, iter_time=iter_time)




