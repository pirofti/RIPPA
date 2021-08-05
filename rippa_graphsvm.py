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

import cvxpy as cp
import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from timeit import default_timer as timer

from rippa import RIPPAEstimator


class RIPPAGraphSVMEstimator(RIPPAEstimator):
    """ A Graph Support Vector Machine estimator with an RIPPA solver. """

    def __init__(self,
                 mu=1,
                 delta=1,
                 epochs=10,
                 ippiters=50,
                 rho=0.05,
                 tau=1,
                 inner='rsm'):
        pparams = {'mu': mu,
                   'delta': delta,
                   'epochs': epochs,
                   'ippiters': ippiters,
                   'rho': rho,
                   'inner': inner, }
        super(RIPPAGraphSVMEstimator, self).__init__(**pparams)

        self.tau = tau
        self.w_opt = None
        self.w_sol = None

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.w0 = kwargs['w0']
        self.C = kwargs['C']
        self.F = kwargs['F']

        self.m, self.n = X.shape

        self._cvxopt(X, y)

        self.update_stats(X, y, self.w0)
        self.stats[0]['time'] = 0
        self.w_sol = self.ipp_restart(X, y, self.w0)

        self.is_fitted_ = True
        return self

    def inner_rsm_(self, X, y, w):
        # Regularized Subgradient Method
        z = w
        self.conv_inner = False
        for k in range(self.K_in):
            z = z - self.alpha * (
                (1/self.n)*X.T@np.sign(np.maximum(0, X@z - y)) +
                self.tau*self.F.T@np.sign(self.F@z) +
                (1/self.mu) * (z - w))
            self.update_stats(X, y, z)
            if self.conv_inner:
                break

        return z

    def fobj_(self, X, y, w):
        r = np.maximum(0, X[:self.m, :]@w - y[:self.m])
        err = np.mean(r) + self.tau * np.linalg.norm(self.F@w, ord=1)
        if self.w_opt is None:
            return err
        errx = np.linalg.norm(w - self.w_opt) ** 2
        return err, errx

    def _cvxopt(self, X, y):
        w = cp.Variable(shape=self.n)
        z = cp.Variable(shape=self.m)
        d = cp.Variable(shape=self.m)
        cost = -(1/self.m)*cp.sum(d) - self.tau*cp.norm(self.F@w, p=1)
        objective = cp.Maximize(cost)
        constraints = [
            z == X@w - y,
            d >= z,
            d >= 0
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        # print('Solver status: {}'.format(prob.status))
        self.f_opt = -prob.solve()
        self.w_opt = w.value
