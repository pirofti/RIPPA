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
from sklearn.base import BaseEstimator
import sys
from timeit import default_timer as timer


class RIPPAEstimator(BaseEstimator):

    def __init__(self,
                 mu=0.1,
                 delta=0.1,
                 epochs=9,
                 ippiters=20,
                 rho=2,
                 crit_stop=0.5,
                 inner='rsm'):
        self.mu = mu
        self.delta = delta
        self.delta_n = self.delta
        self.epochs = epochs
        self.ippiters = ippiters
        self.rho = rho
        self.inner = inner

        self.stats = []
        self.start = 0
        self.do_stats = True

        self.crit_stop = crit_stop
        self.conv_inner = False
        self.conv_outer = False
        self.conv_epoch = False

        self.inner_ = self.inner_rsm_
        self.inner_iters_ = self.inner_iters_rsm_
        self.params_init_ = self.params_init_rsm_
        self.params_update_ = self.params_update_rsm_
        self.find_delta0 = self.find_delta0_1_inner

    def ipp_restart(self, X, y, w0):
        self.params_init_(X, y, w0)

        self.epoch = 0
        self.start = timer()
        # for self.epoch in range(self.epochs):
        while not self.conv_inner:
            self.inner_iters_()

            w_sol = self.ipp(X, y, w0)

            # Update params
            self.params_update_()
            self.epoch = self.epoch + 1

            # See if sigmaF reached
            self.ipp_last(X, y, w0)

            w0 = w_sol

        return w_sol

    def find_delta0_1_inner(self, X, y, w0):
        self.do_stats = False
        itd0 = 0
        K = 10
        crit_stop = self.delta_n + 1
        self.K_in = 1
        w = self.inner_(X, y, w0)
        self.delta_n = np.linalg.norm(w - w0) / self.mu
        self.delta = self.delta_n * self.mu
        self.do_stats = True

    def find_delta0_n_inner(self, X, y, w0):
        self.do_stats = False
        itd0 = 0
        K = 10
        crit_stop = self.delta_n + 1
        while crit_stop > self.delta_n:
            self.K_in = np.maximum(100, int(
                np.round(8 * np.log(self.lmax / self.delta_n))))
            K = int(np.round(K * 2**(0.2 * itd0)))
            w = w0

            for k in range(K):
                w = self.inner_(X, y, w)

            crit_stop = np.linalg.norm(w - w0)
            self.delta_n = 2 * self.delta_n
            itd0 = itd0+1
        self.delta = self.delta_n*self.mu
        self.do_stats = True

    def find_delta0_lmax(self, X, y, w0):
        self.delta_n = 2 * self.lmax

    def params_init_rsm_(self, X, y, w0):
        EIGTHRES = 1e-6
        [m, n] = X.shape

        # Compute number of iterations
        G = X@X.T
        l, _ = np.linalg.eigh(G)
        l[l < EIGTHRES] = 0
        self.lmin = np.amin(l[np.nonzero(l)])
        self.lmax = np.amax(l)
        self.alpha = self.mu/2
        self.find_delta0(X, y, w0)
        self.delta = self.mu * self.delta_n

    def inner_iters_rsm_(self):
        K = np.maximum(100, (self.rho-1)*2**(2*self.rho*self.epoch))
        self.K_in = int(np.round(K))

    def params_update_rsm_(self):
        self.mu = self.mu * 2
        self.alpha = self.alpha / 2**(2*self.rho - 1)
        self.delta_n = self.delta_n / (2 ** self.rho)
        self.delta = self.mu * self.delta_n

    def params_update_rsm_last_(self):
        self.alpha = self.alpha / 2

    def update_stats(self, X, y, w):
        if self.do_stats is False:
            return
        titer = timer()
        err, errx = self.fobj_(X, y, w)
        self.stats.append({"x": w, "xdist": errx, "error": err,
                           "time": titer - self.start})
        # print(f'conv_inner: {err - self.f_opt} <= {self.crit_stop}')
        if err - self.f_opt <= self.crit_stop:
            self.conv_inner = True

    def ipp(self, X, y, w0):
        w = w0

        ippiter = 0
        self.conv_outer = False
        self.conv_inner = False
        eps_stop = self.delta

        start = timer()
        # for ippiter in range(self.ippiters):
        while not self.conv_outer:
            w_old = w
            w = self.inner_(X, y, w)
            crit_stop = np.linalg.norm(w - w_old)
            # print(f'conv_outer: {crit_stop} <= {eps_stop}')
            self.conv_outer = (crit_stop <= eps_stop) or self.conv_inner
            ippiter = ippiter + 1

        return w

    def ipp_last(self, X, y, w0):
        w = w0

        ippiter = 0
        self.conv_inner = False
        self.K_in = round((self.lmax / self.delta_n)**2)
        ippiters = 2*np.log(self.lmax / (2*self.delta_n*self.crit_stop))
        ippiters = round(ippiters)
        self.params_save()
        for ippiter in range(ippiters):
            w_old = w
            w = self.inner_(X, y, w)
            self.params_update_rsm_last_()
            ippiter = ippiter + 1
        self.params_restore()

        return w

    def params_save(self):
        self.alpha_saved = self.alpha
        self.delta_n_saved = self.delta_n
        self.delta_saved = self.delta

    def params_restore(self):
        self.alpha= self.alpha_saved
        self.delta_n= self.delta_n_saved
        self.delta= self.delta_saved
