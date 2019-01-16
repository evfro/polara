from math import sqrt
import numpy as np
from numba import jit, njit, prange

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
def adagrad(grad, cum_sq_grad, smoothing=1e-6):
    cum_sq_grad += grad * grad
    adjusted_grad = grad / (smoothing + np.sqrt(cum_sq_grad))
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

        adjusted_ngrad_p = adjust_gradient(ngrad_p, *adjustment_params)
        new_pm = pm + eta * adjusted_ngrad_p
        adjusted_ngrad_q = adjust_gradient(ngrad_q, *adjustment_params)
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
                       seed=None, verbose=False, iter_errors=None):
    sgd_sweep_func = sgd_sweep_func or generalized_sgd_sweep
    apply_kernel = apply_kernel or identity
    kernel_params = kernel_params or ((), ())
    adjust_gradient = adjust_gradient or identity
    adjustment_params = adjustment_params or ()

    assert isinstance(interactions, tuple) # required by numba
    assert isinstance(nonzero_count, tuple) # required by numba

    rnds = np.random if seed is None else np.random.RandomState(seed)
    row_factors = rnds.normal(scale=0.1, size=(shape[0], rank))
    col_factors = rnds.normal(scale=0.1, size=(shape[1], rank))

    nnz = len(interactions[-1])
    last_err = np.finfo(np.float64).max
    for epoch in range(num_epochs):
        if adjust_gradient is adagrad:
            adjustment_params = (np.zeros(rank, dtype='f8'),)

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
                   seed=None, verbose=False, iter_errors=None):
    lambd = 0.5 * sigma**2
    return mf_sgd_boilerplate(interactions, shape, nonzero_count, rank,
                              lrate, lambd, num_epochs, tol,
                              adjust_gradient=adjust_gradient,
                              adjustment_params=adjustment_params,
                              seed=seed, verbose=verbose, iter_errors=iter_errors)



