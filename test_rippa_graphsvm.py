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
import scipy

from rippa_graphsvm import RIPPAGraphSVMEstimator
from rippa_util import f_choose_min


def rippa_rho():
    t_rippa = []
    rhos = np.arange(1.005, 1.1, 0.005)
    for rho in rhos:
        print(f'Starting rho={rho}')
        params['rho'] = rho
        ipp_rsm = RIPPAGraphSVMEstimator(**params, inner='rsm')
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
    plt.savefig("figs/rippa-graphsvm-rho.pdf")


def rippa_mn():
    t_rippa = []
    dims = np.arange(10, 200, 5)
    for n_dim in dims:
        C = np.random.standard_normal((n_dim, n_dim))
        labels = 2 * (np.random.standard_normal(n_dim) > 0.5) - 1

        # SVM to Robust L1LS
        X = -np.diag(labels)@C.T
        y = -np.ones(labels.shape)

        F = np.random.standard_normal((X.shape[1], X.shape[1]))

        w0 = np.random.standard_normal(X.shape[1])
        data_params = {'w0': w0, 'C': C, 'F': F}

        print(f'Starting m={n_dim}')
        ipp_rsm = RIPPAGraphSVMEstimator(**params, inner='rsm')
        ipp_rsm.fit(X, y, **data_params)
        print(f'm={n_dim}: {np.size(ipp_rsm.stats)} iters')
        t_rippa.append(np.size(ipp_rsm.stats))

    f = open('data/rippa-graphsvm-mn.p', 'wb')
    pickle.dump(t_rippa, f)
    pickle.dump(dims, f)
    f.close()

    plt.figure()
    plt.xlabel(r'$m=n$')
    plt.ylabel('Total number of inner iterations')
    tz = dims
    m_label = r'RIPPA($m=n$)'
    # plt.xticks(dims)
    plt.plot(tz, t_rippa, 'go', label=m_label)
    plt.legend()
    plt.show()
    plt.savefig("figs/rippa-graphsvm-mn.pdf")


def rippa_tau():
    t_rippa = []
    taus = np.arange(0, 20, 0.1)
    for tau in taus:
        print(f'Starting tau={tau}')
        params['tau'] = tau
        ipp_rsm = RIPPAGraphSVMEstimator(**params, inner='rsm')
        ipp_rsm.fit(X, y, **data_params)
        print(f'tau={tau}: {np.size(ipp_rsm.stats)} iters')
        t_rippa.append(np.size(ipp_rsm.stats))

    f = open('data/rippa-graphsvm-tau.p', 'wb')
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
    plt.savefig("figs/rippa-graphsvm-tau.pdf")


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
    plt.savefig("figs/ipp-graphsvm-time-inner.pdf")


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
    plt.savefig("figs/ipp-graphsvm-iters-inner.pdf")


def rippa_error():
    ipp_rsm = RIPPAGraphSVMEstimator(**params, inner='rsm')
    ipp_rsm.fit(X, y, **data_params)

    pickle.dump(ipp_rsm.stats, f)

    plot_time_inner(ipp_rsm)
    plot_iters_inner(ipp_rsm)


if __name__ == "__main__":
    n_components = 100
    n_features = 512
    tau = 10

    params = {'mu': 0.1, 'rho': 1.0005, 'tau': tau, }

    graphsvm = 1
    l1svm = 0

    C = np.random.standard_normal((n_components, n_features))
    labels = 2 * (np.random.standard_normal(n_features) > 0.5) - 1

    # SVM to Robust L1LS
    X = -np.diag(labels)@C.T
    y = -np.ones(labels.shape)

    if graphsvm:
        # F = scipy.io.loadmat('data/graphsvm.mat')['F']          # 20newsgroups
        F = np.random.standard_normal((X.shape[1], X.shape[1]))   # synthetic
        f = open('data/ipp-graphsvm-new.p', 'wb')
    elif l1svm:
        F = np.eye(X.shape[1])
        f = open('data/ipp-l1svm.p', 'wb')

    w0 = np.random.standard_normal(X.shape[1])

    data_params = {'w0': w0, 'C': C, 'F': F}

    # rippa_rho()
    # rippa_mn()
    # rippa_tau()
    rippa_error()

    f.close()
