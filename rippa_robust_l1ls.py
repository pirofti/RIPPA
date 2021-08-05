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

import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y

from rippa import RIPPAEstimator
from simplex_projection import euclidean_proj_l1ball


class RIPPARobustL1LSEstimator(RIPPAEstimator):
    """ A Robust L1 Least Squares estimator with an RIPPA solver. """

    def __init__(self,
                 mu=0.1,
                 delta=0.1,
                 epochs=9,
                 ippiters=20,
                 rho=2,
                 tau=1,
                 inner='rsm'):
        pparams = {'mu': mu,
                   'delta': delta,
                   'epochs': epochs,
                   'ippiters': ippiters,
                   'rho': rho,
                   'inner': inner}
        super().__init__(**pparams)

        self.tau = tau
        self.w_opt = None
        self.w_sol = None

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.m, self.n = X.shape
        self.w0 = kwargs['w0']

        self._cvxopt(X, y)
        self.w_sol = self.ipp_restart(X, y, self.w0)

        self.is_fitted_ = True
        return self

    def inner_rsm_(self, X, y, w):
        # Regularized Subgradient Method
        z = w
        for k in range(self.K_in):
            z = z - self.alpha * (X.T@np.sign(X@z - y) + 1/self.mu * (z - w))
            z = euclidean_proj_l1ball(z, s=self.tau)
            self.update_stats(X, y, z)

        return z

    def fobj_(self, X, y, w):
        err = np.linalg.norm(X@w - y, ord=1)
        if self.w_opt is None:
            return err
        errx = np.linalg.norm(w - self.w_opt) ** 2
        return err, errx

    def _cvxopt(self, X, y):
        n = X.shape[1]
        w = cp.Variable(shape=n)
        cost = -cp.norm(X @ w - y, p=1)
        objective = cp.Maximize(cost)
        constraints = [cp.norm(w, p=1) <= self.tau]
        prob = cp.Problem(objective, constraints)
        # prob = cp.Problem(objective)
        self.f_opt = -prob.solve()
        self.w_opt = w.value
