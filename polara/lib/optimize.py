from math import sqrt
import numpy as np
from numba import jit, njit, prange
from scipy import sparse

from polara.tools.timing import track_time


@njit(nogil=True)
def mf_sgd_sweep(users_idx, items_idx, feedbacks, P, Q, eta, lambd, *args,
                 adjust_gradient, adjustment_params):
    cum_error = 0
    for k, a in enumerate(feedbacks):
        i = users_idx[k]
        j = items_idx[k]

        pi = P[i, :]
        qj = Q[j, :]

        err = a - pi @ qj

        ngrad_p = err*qj - lambd*pi
        adjusted_ngrad_p = adjust_gradient(ngrad_p, i, *adjustment_params[0])
        new_pi = pi + eta * adjusted_ngrad_p

        ngrad_q = err*pi - lambd*qj
        adjusted_ngrad_q = adjust_gradient(ngrad_q, j, *adjustment_params[1])
        new_qj = qj + eta * adjusted_ngrad_q

        P[i, :] = new_pi
        Q[j, :] = new_qj

        cum_error += err*err
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

@njit(nogil=True)
def gnpropz(grad, m, cum_sq_norm, smoothing=1e-6):
    cum_sq_norm_update = cum_sq_norm[m] + grad @ grad
    cum_sq_norm[m] = cum_sq_norm_update
    adjusted_grad = grad / sqrt(smoothing + cum_sq_norm_update)
    return adjusted_grad


@njit(nogil=True, parallel=False)
def generalized_sgd_sweep(row_idx, col_idx, values, P, Q,
                          eta, lambd, row_nnz, col_nnz,
                          transform, transform_params,
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

        kpm = transform(pm, P, m, *transform_params[0])
        ngrad_p = err * qn - kpm * row_lambda
        sqn = transform(qn, Q, n, *transform_params[1])
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
                       transform=None, transform_params=None,
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
    transform = transform or identity
    transform_params = transform_params or ((), ())
    adjust_gradient = adjust_gradient or identity
    adjustment_params = adjustment_params or ((), ())

    nnz = len(interactions[-1])
    last_err = np.finfo('f8').max
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
                                     transform, transform_params,
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


def simple_mf_sgd(interactions, shape, nonzero_count, rank,
                  lrate, lambd, num_epochs, tol,
                  adjust_gradient=None, adjustment_params=None,
                  seed=None, verbose=False,
                  iter_errors=None, iter_time=None):
    #nonzero_count = ((), ())
    nonzero_count = (np.ones(shape[0]), np.ones(shape[1]))
    return mf_sgd_boilerplate(interactions, shape, nonzero_count, rank,
                              lrate, lambd, num_epochs, tol,
                              adjust_gradient=adjust_gradient,
                              adjustment_params=adjustment_params,
                              sgd_sweep_func=generalized_sgd_sweep,
                              seed=seed, verbose=verbose,
                              iter_errors=iter_errors, iter_time=iter_time)


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


def sp_kernel_update(pm, P, m, K):
    k = K.getrow(m)
    kp = k.dot(P).squeeze()
    return kp + k[0, m] * pm

@njit(nogil=True, parallel=False)
def sparse_kernel_update(pm, P, m, kernel_ptr, kernel_ind, kernel_data):
    lead_idx = kernel_ptr[m]
    stop_idx = kernel_ptr[m+1]

    kernel_update = np.zeros_like(pm)

    for i in range(lead_idx, stop_idx):
        index = kernel_ind[i]
        value = kernel_data[i]
        p_row = P[index, :]
        if index == m: # diagonal value
            p_row = p_row + pm # avoid rewriting original data
        kernel_update += value * p_row
    return kernel_update


def kernelized_pmf_sgd(interactions, shape, nonzero_count, rank,
                       lrate, sigma, num_epochs, tol,
                       kernel_matrices, kernel_update=None, sparse_kernel_format=True,
                       adjust_gradient=None, adjustment_params=None,
                       seed=None, verbose=False, iter_errors=None, iter_time=None):
    kernel_update = kernel_update or sparse_kernel_update

    row_kernel, col_kernel = kernel_matrices
    if sparse_kernel_format:
        row_kernel_data = (row_kernel.indptr, row_kernel.indices, row_kernel.data)
        col_kernel_data = (col_kernel.indptr, col_kernel.indices, col_kernel.data)
    else:
        row_kernel_data = (row_kernel,)
        col_kernel_data = (col_kernel,)

    kernel_params = (row_kernel_data, col_kernel_data)

    lambd = 0.5 * sigma**2
    return mf_sgd_boilerplate(interactions, shape, nonzero_count, rank,
                              lrate, lambd, num_epochs, tol,
                              sgd_sweep_func=generalized_sgd_sweep,
                              transform=kernel_update,
                              transform_params=kernel_params,
                              adjust_gradient=adjust_gradient,
                              adjustment_params=adjustment_params,
                              seed=seed, verbose=verbose,
                              iter_errors=iter_errors, iter_time=iter_time)


def trace(A, B):
    if sparse.issparse(A):
        return A.multiply(B).sum()
    return (A * B).sum()

def local_collective_embeddings(Xs, Xu, A, k=15, alpha=0.1, beta=0.05,
                                lamb=1, epsilon=0.0001, maxiter=15,
                                seed=None, verbose=True):
    """
    Python Implementation of Local Collective Embeddings

    author : Abhishek Thakur, https://github.com/abhishekkrthakur/LCE
    original : https://github.com/msaveski/LCE
    adapted for Polara by: Evgeny Frolov
    """
    n = Xs.shape[0]
    v1 = Xs.shape[1]
    v2 = Xu.shape[1]

    random = np.random if seed is None else np.random.RandomState(seed)
    W = random.rand(n, k)
    Hs = random.rand(k, v1)
    Hu = random.rand(k, v2)

    D = sparse.dia_matrix((A.sum(axis=0), 0), A.shape)

    gamma = 1. - alpha
    trXstXs = trace(Xs, Xs)
    trXutXu = trace(Xu, Xu)

    WtW = W.T.dot(W)
    WtXs = Xs.T.dot(W).T
    WtXu = Xu.T.dot(W).T
    WtWHs = WtW.dot(Hs)
    WtWHu = WtW.dot(Hu)
    DW = D.dot(W)
    AW = A.dot(W)

    itNum = 1
    delta = 2.0 * epsilon

    ObjHist = []

    while True:

        # update H
        Hs_1 = np.divide(
            (alpha * WtXs), np.maximum(alpha * WtWHs + lamb * Hs, 1e-10))
        Hs = np.multiply(Hs, Hs_1)

        Hu_1 = np.divide(
            (gamma * WtXu), np.maximum(gamma * WtWHu + lamb * Hu, 1e-10))
        Hu = np.multiply(Hu, Hu_1)

        # update W
        W_t1 = alpha * Xs.dot(Hs.T) + gamma * Xu.dot(Hu.T) + beta * AW
        W_t2 = alpha * W.dot(Hs.dot(Hs.T)) + gamma * \
            W.dot(Hu.dot(Hu.T)) + beta * DW + lamb * W
        W_t3 = np.divide(W_t1, np.maximum(W_t2, 1e-10))
        W = np.multiply(W, W_t3)

        # calculate objective function
        WtW = W.T.dot(W)
        WtXs = Xs.T.dot(W).T
        WtXu = Xu.T.dot(W).T
        WtWHs = WtW.dot(Hs)
        WtWHu = WtW.dot(Hu)
        DW = D.dot(W)
        AW = A.dot(W)

        tr1 = alpha * (trXstXs - 2. * trace(Hs, WtXs) + trace(Hs, WtWHs))
        tr2 = gamma * (trXutXu - 2. * trace(Hu, WtXu) + trace(Hu, WtWHu))
        tr3 = beta * (trace(W, DW) - trace(W, AW))
        tr4 = lamb * (np.trace(WtW) + trace(Hs, Hs) + trace(Hu, Hu))

        Obj = tr1 + tr2 + tr3 + tr4
        ObjHist.append(Obj)

        if itNum > 1:
            delta = abs(ObjHist[-1] - ObjHist[-2])
            if verbose:
                print("Iteration: ", itNum, "Objective: ", Obj, "Delta: ", delta)
            if itNum > maxiter or delta < epsilon:
                break

        itNum += 1

    return W, Hu, Hs
