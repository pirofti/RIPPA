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
import pickle
from sklearn.datasets import make_sparse_coded_signal

from rippa_robust_l1ls import RIPPARobustL1LSEstimator
from rippa_util import f_choose_min
from simplex_projection import euclidean_proj_l1ball


def rippa_rho():
    t_rippa = []
    rhos = np.arange(1.005, 1.1, 0.005)
    for rho in rhos:
        print(f'Starting rho={rho}')
        params['rho'] = rho
        ipp_rsm = RIPPARobustL1LSEstimator(**params, inner='rsm')
        ipp_rsm.fit(X, y, **data_params)
        t_rippa.append(np.size(ipp_rsm.stats))

    plt.figure()
    plt.xlabel(r'$\rho$')
    plt.ylabel('Total number of inner iterations')
    tz = rhos
    m_label = r'RIPPA($\rho$)'
    plt.plot(tz, t_rippa, 'g', label=m_label)
    plt.legend()
    plt.show()
    plt.savefig("figs/rippa-rho.pdf")


def rippa_mn():
    t_rippa = []
    dims = np.arange(10, 200, 5)
    for n_dim in dims:
        # Generate sparse signals using a dictionary
        y, X, w_truth = make_sparse_coded_signal(
            n_samples=n_samples,
            n_components=n_dim,
            n_features=n_dim,
            n_nonzero_coefs=n_nonzero_coefs,
            random_state=0)
        w0 = np.random.standard_normal(X.shape[1])
        data_params = {'w0': w0}

        print(f'Starting m={n_dim}')
        ipp_rsm = RIPPARobustL1LSEstimator(**params, inner='rsm')
        ipp_rsm.fit(X, y, **data_params)
        print(f'm={n_dim}: {np.size(ipp_rsm.stats)} iters')
        t_rippa.append(np.size(ipp_rsm.stats))

    f = open('data/rippa-mn.p', 'wb')
    pickle.dump(t_rippa, f)
    pickle.dump(dims, f)
    f.close()

    plt.figure()
    plt.xlabel(r'$m=n$')
    plt.ylabel('Total number of inner iterations')
    tz = dims
    m_label = r'RIPPA($m=n$)'
    plt.plot(tz, t_rippa, 'go', label=m_label)
    plt.legend()
    plt.show()
    plt.savefig("figs/rippa-mn.pdf")


def rippa_tau():
    t_rippa = []
    taus = np.arange(0.1, 0.5, 0.1)
    for tau in taus:
        print(f'Starting tau={tau}')
        params['tau'] = tau
        ipp_rsm = RIPPARobustL1LSEstimator(**params, inner='rsm')
        ipp_rsm.fit(X, y, **data_params)
        t_rippa.append(np.size(ipp_rsm.stats))

    f = open('data/rippa-tau.p', 'wb')
    pickle.dump(t_rippa, f)
    pickle.dump(taus, f)
    f.close()

    plt.figure()
    plt.xlabel(r'$\tau$')
    plt.ylabel('Total number of inner iterations')
    tz = taus
    m_label = r'RIPPA($\tau$)'
    plt.plot(tz, t_rippa, 'g', label=m_label)
    plt.legend()
    plt.show()
    plt.savefig("figs/rippa-tau.pdf")


def plot_time_inner(ipp_rsm):
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
    plt.savefig("figs/ipp-robustl1ls-time-inner.pdf")


def plot_iters_inner(ipp_rsm):
    plt.figure()
    ax = plt.gca()
    # ax.set_title('L1-SVM')
    plt.xlabel('Inner Iterations')
    plt.ylabel('$f - f^\star$')

    tz = np.arange(np.size(ipp_rsm.stats))
    errz_rsm = [i['error'] for i in ipp_rsm.stats]
    errz_rsm = errz_rsm - ipp_rsm.f_opt
    errz_rsm = f_choose_min(np.array(errz_rsm))
    m_label = 'RIPPA'
    plt.semilogy(tz, errz_rsm, 'g', label=m_label)

    plt.legend()
    plt.show()
    plt.savefig("figs/ipp-robustl1ls-iters-inner.pdf")


def rippa_error():
    ipp_rsm = RIPPARobustL1LSEstimator(**params, inner='rsm')
    ipp_rsm.fit(X, y, **data_params)

    f = open('data/ipp-l1svm.p', 'wb')
    pickle.dump(ipp_rsm.stats, f)
    f.close()

    plot_time_inner(ipp_rsm)
    plot_iters_inner(ipp_rsm)


if __name__ == "__main__":
    n_nonzero_coefs = 3     # sparsity (s)
    n_samples = 1           # number of signals (N)
    n_components = 8
    n_features = 1000
    tau = 1

    params = {'mu': 0.1, 'rho': 1.005, 'tau': tau, }

    # Generate sparse signals using a dictionary
    y, X, w_truth = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=0)
    w0 = np.random.standard_normal(X.shape[1])
    w0 = euclidean_proj_l1ball(w0, s=tau)

    data_params = {'w0': w0}

    # rippa_rho()
    # rippa_mn()
    # rippa_tau()
    rippa_error()
