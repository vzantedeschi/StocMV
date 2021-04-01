#!/usr/bin/env python
import sys
import argparse
import torch
import numpy as np
import cvxpy as cp

from sklearn.base import BaseEstimator, ClassifierMixin


###############################################################################

class MajorityVote(BaseEstimator, ClassifierMixin):

    def __init__(self, X, y, complemented=False, quasi_uniform=False):
        assert (isinstance(complemented, bool)
                and isinstance(quasi_uniform, bool))

        self.X = X
        self.y = y

        self.complemented = complemented
        self.quasi_uniform = quasi_uniform

        self.prior = None
        self.post = None
        self.X_y_list = (np.array(X), np.array(y))
        self.fitted = False

    def fit(self):
        self.fitted = True
        self.voter_list = []

        return self

    def output(self, X):
        out = None

        for voter in self.voter_list:

            if(out is None):
                out = voter.output(X)
            elif(isinstance(X, torch.Tensor)):
                out = torch.cat((out, voter.output(X)), dim=1)
            else:
                out = np.concatenate((out, voter.output(X)), 1)
        return out

    def predict(self, X):
        out = self.output(X)
        pred = out@self.post
        return pred

    def quasi_uniform_to_normal(self):
        assert self.complemented
        assert self.quasi_uniform

        # We consider that the posterior is obtained through X
        X, _ = self.X_y_list
        out = self.output(X)

        post_ = cp.Variable(shape=self.post.shape)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(post_, self.prior))),
                          [out@post_ == out@self.post,
                           cp.sum(post_) == 1,
                           post_ >= 0])
        prob.solve()
        self.post = np.abs(post_.value)/np.sum(np.abs(post_.value))
        self.quasi_uniform = False

    def normal_to_quasi_uniform(self):
        assert self.complemented
        assert not(self.quasi_uniform)

        # We applied Theorem 43 of
        # "Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis
        # to a Learning Algorithm" from Germain et al. (2015)
        # to convert the post. for "normal" into "quasi-uniform" posterior
        n = len(self.voter_list)
        post_n = self.post[:n//2]
        post_2n = self.post[n//2:]
        post_n_ = (1/n) - (post_n-post_2n)/(n*np.max(np.abs(post_n-post_2n)))
        post_2n_ = (1/n) - (post_2n-post_n)/(n*np.max(np.abs(post_n-post_2n)))
        self.post = np.concatenate((post_n_, post_2n_), axis=0)
        self.quasi_uniform = True

    def switch_complemented(self):
        assert self.quasi_uniform

        n = len(self.voter_list)

        if(self.complemented):
            self.complemented = False
            post_n = self.post[:n//2]
            post_2n = self.post[n//2:]
            post = post_n-post_2n
            self.fit()
            self.post = post
        else:
            self.complemented = True
            post_ = 0.5*(self.post+1.0/(n))
            self.fit()
            self.post = np.concatenate((post_, (1.0/n)-post_), axis=0)
