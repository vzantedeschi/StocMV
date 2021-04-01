import logging
import numpy as np
import cvxpy as cp

from sklearn.base import BaseEstimator, ClassifierMixin
from voter.majority_vote import MajorityVote


class MinCqLearnerV2(BaseEstimator, ClassifierMixin):

    def __init__(self, majority_vote, mu):
        assert mu > 0 and mu <= 1
        self.mu = mu
        self.majority_vote = majority_vote
        self.mv = majority_vote
        assert isinstance(self.mv, MajorityVote)
        assert self.mv.fitted
        assert self.mv.complemented
        # Not implemented for quasi-uniform
        assert not(self.mv.quasi_uniform)

    def get_params(self, deep=True):
        return {"mu": self.mu, "majority_vote": self.mv}

    def fit(self, X, y):
        # X -> (size, nb_feature)
        # y -> (size, 1)
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(X.shape) == 2 and len(y.shape) == 2 and
                X.shape[0] == y.shape[0] and
                y.shape[1] == 1 and X.shape[0] > 0)

        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        out = self.mv.output(X)

        nb_voter = len(self.mv.post)
        m = len(X)

        u = 1.0
        l = 0.0

        while(u-l > 0.001):
            t = (l+u)/2.0

            post_ = cp.Variable(shape=(nb_voter, 1))

            vote = out@post_

            mu_1 = (1/m)*cp.sum(cp.multiply(y, vote))
            mu_2 = (1/m)*cp.sum((vote)**2.0)
            prob = cp.Problem(
                cp.Minimize(mu_1**2.0+t*(mu_2)),
                [mu_1 == self.mu, cp.sum(post_) == 1, post_ >= 0])

            try:
                prob.solve()
                post = post_.value
            except cp.error.SolverError:
                post = None

            if(post is None):
                post = self.mv.prior
                break

            mu_1 = mu_1.value
            mu_2 = mu_2.value

            c_bound = 1.0-((mu_1)**2.0)/(mu_2)

            if(c_bound > 1.0-t):
                u = t
            else:
                l = t

        self.mv.post = post
        return self

    def predict(self, X):
        return self.mv.predict(X)
