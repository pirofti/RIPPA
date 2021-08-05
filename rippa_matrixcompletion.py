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
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

from rippa import RIPPAEstimator


class RIPPAMatrixCompletionEstimator(RIPPAEstimator):
    """ A Matrix Completion estimator with an RIPPA solver. """

    def __init__(self,
                 mu=1,
                 delta=1,
                 epochs=10,
                 ippiters=50,
                 rho=0.05,
                 tau=1,
                 crit_stop=0.5,
                 inner='rsm'):
        pparams = {'mu': mu,
                   'delta': delta,
                   'epochs': epochs,
                   'ippiters': ippiters,
                   'rho': rho,
                   'crit_stop': crit_stop,
                   'inner': inner, }
        super().__init__(**pparams)

        self.tau = tau
        self.w_opt = None
        self.w_sol = None
        self.inner = inner

        self.f_opt = 0
        self.w_opt = 0

        self.inner_ = self.inner_rsm_

    def fit(self, X, y, **kwargs):
        self.w0 = kwargs['w0']
        self.ind = kwargs['ind']

        self.m, self.n = self.w0.shape
        self.N = y.shape[0]
        self.gw = np.zeros_like(self.w0)

        self._cvxopt(X, y)

        self.w_sol = self.ipp_restart(X, y, self.w0)

        self.is_fitted_ = True
        return self

    def inner_rsm_(self, X, y, w):
        # Regularized Subgradient Method
        z = w
        self.conv_inner = False
        for k in range(self.K_in):
            U, _, V = np.linalg.svd(z, full_matrices=False)
            self.gw.flat[self.ind] = np.sign(z.flat[self.ind] - y)
            z = z - self.alpha * (
                (1/self.N) * self.gw +
                self.tau * U@V +
                1/self.mu * (z - w))
            self.update_stats(X, y, z)
            if self.conv_inner:
                break

        return z

    def fobj_(self, X, y, w):
        loss = np.mean(np.absolute(w.flat[self.ind] - y))
        reg = np.linalg.norm(w, ord='nuc')
        err = loss + self.tau * reg
        # print(f"loss = {loss} reg = {reg}")
        if self.w_opt is None:
            return err, 0
        errx = np.linalg.norm(w - self.w_opt) ** 2
        return err, errx

    def _cvxopt(self, X, y):
        # Build ratings matrix
        ym = np.zeros_like(self.w0)
        rows, cols = np.unravel_index(self.ind, ym.shape)
        ym[rows, cols] = y

        w = cp.Variable(shape=self.w0.shape)
        cost = -(1/self.N)*cp.sum(cp.abs(w - ym)) - self.tau*cp.normNuc(w)
        objective = cp.Maximize(cost)
        prob = cp.Problem(objective)
        prob.solve()
        print('Solver status: {}'.format(prob.status))
        self.f_opt = -prob.solve()
        self.w_opt = w.value
