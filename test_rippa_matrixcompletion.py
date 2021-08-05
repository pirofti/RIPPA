# Copyright (c) 2021 Paul Irofti <paul@irofti.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_sparse_coded_signal

from rippa_matrixcompletion import RIPPAMatrixCompletionEstimator
from rippa_util import f_choose_min


def plot_time_inner():
    plt.figure()
    ax = plt.gca()
    plt.xlabel('Inner iterations CPU Time (s)')
    plt.ylabel('$f - f^\star$')

    tz = [i['time'] for i in ipp_rsm.stats]
    errz_rsm = [i['error'] for i in ipp_rsm.stats]
    errz_rsm = errz_rsm - ipp_rsm.f_opt
    errz_rsm = f_choose_min(np.array(errz_rsm))
    m_label = 'RIPPA'
    plt.semilogy(tz, errz_rsm, 'g', label=m_label)

    plt.legend()
    plt.show()
    plt.savefig("figs/ipp-mc-time-inner.pdf")


def plot_iters_inner():
    plt.figure()
    ax = plt.gca()
    plt.xlabel('Number of Inner Iterations')
    plt.ylabel('$f - f^\star$')

    tz = np.arange(np.size(ipp_rsm.stats))
    errz_rsm = [i['error'] for i in ipp_rsm.stats]
    errz_rsm = errz_rsm - ipp_rsm.f_opt
    errz_rsm = f_choose_min(np.array(errz_rsm))
    m_label = 'RIPPA'
    plt.semilogy(tz, errz_rsm, 'g', label=m_label)

    plt.legend()
    plt.show()
    plt.savefig("figs/ipp-mc-iters-inner.pdf")


if __name__ == "__main__":
    tau = 3

    params = {'mu': 0.1, 'rho': 1.005, 'crit_stop': 0.01, 'tau': tau, }

    n_movies = 100
    n_users = 40
    n_nonzero_coefs = int(n_users / 4)
    _, _, ratings = make_sparse_coded_signal(
        n_samples=n_movies,
        n_components=n_users,
        n_features=n_users,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=0)
    user_ids, movie_ids = np.nonzero(ratings)
    ind = np.ravel_multi_index((user_ids, movie_ids), (n_users, n_movies))
    y = np.random.randint(1, high=6, size=len(ind))
    w0 = np.random.standard_normal(ratings.shape)

    X = np.eye(n_users)     # not used

    data_params = {'w0': w0, 'ind': ind}

    ipp_rsm = RIPPAMatrixCompletionEstimator(**params, inner='rsm')
    ipp_rsm.fit(X, y, **data_params)

    plot_time_inner()
    plot_iters_inner()
