import numpy as np
from numba import njit

@njit(nogil=True)
def sgd_step(users_idx, items_idx, feedbacks, P, Q, eta, lambd):
    cum_error = 0
    for k, a in enumerate(feedbacks):
        i = users_idx[k]
        j = items_idx[k]

        pi = P[i, :]
        qj = Q[j, :]

        e = a - np.dot(pi, qj)

        new_pi = pi + eta * (e*qj - lambd*pi)
        new_qj = qj + eta * (e*pi - lambd*qj)

        P[i, :] = new_pi
        Q[j, :] = new_qj

        cum_error += e*e
    return cum_error

@njit(nogil=True)
def sgd_step_biased(users_idx, items_idx, feedbacks, P, Q, b_user, b_item, mu, eta, lambd):
    cum_error = 0
    for k, a in enumerate(feedbacks):
        i = users_idx[k]
        j = items_idx[k]

        pi = P[i, :]
        qj = Q[j, :]
        bi = b_user[i]
        bj = b_item[j]

        e = a - (np.dot(pi, qj) + bi + bj + mu)

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
