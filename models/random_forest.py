import torch
import numpy as np
import random

import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from voter.majority_vote import MajorityVote
from sklearn.utils import check_random_state

###############################################################################

class Tree(BaseEstimator, ClassifierMixin):

    def __init__(self, dir, rand=None):
        self.tree = DecisionTreeClassifier(
            criterion="gini",
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            random_state=rand)
        self.dir = dir


    def fit(self, X, y):
        self.tree.fit(X,y)
        return self

    def output(self, X):
        # X -> (size, nb_feature)
        assert ((isinstance(X, torch.Tensor) or isinstance(X, np.ndarray))
                and (len(X.shape)== 2))

        X_ = X
        if(isinstance(X, torch.Tensor)):
            X_ = X.detach().cpu().numpy()

        out = self.dir*np.expand_dims(self.tree.predict(X_), 1)

        if(isinstance(X, torch.Tensor)):
            out = torch.tensor(out, device=X.device)
        return out


class TreeMV(MajorityVote):

    def __init__(
        self, X, y,
        nb_tree=100, complemented=False, quasi_uniform=False
    ):
        self.nb_tree = nb_tree

        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(X.shape) == 2 and len(y.shape) == 2 and
                X.shape[0] == y.shape[0] and
                y.shape[1] == 1 and X.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        super().__init__(
            X, y, complemented=complemented, quasi_uniform=quasi_uniform)

        # Same as Masegosa et al.
        self.seed = 1000
        self.fit()

    def fit(self):
        super().fit()
        X, y = self.X_y_list

        y_list = [+1]
        if(self.complemented):
            y_list = [+1, -1]

        y_ = y_list[0]
        for i in range(self.nb_tree):
            self.voter_list.append(Tree(y_, rand=i))
            self.voter_list[i].fit(X, y)
        if(self.complemented):
            y_ = y_list[1]
            for i in range(self.nb_tree):
                self.voter_list.append(Tree(y_, rand=i))
                self.voter_list[self.nb_tree+i].fit(X, y)

        if(not(self.complemented) and self.quasi_uniform):
            self.prior = np.zeros((len(self.voter_list), 1))
        else:
            self.prior = ((1.0/len(self.voter_list))
                          *np.ones((len(self.voter_list), 1)))

        self.post = np.array(self.prior)

        return self

###############################################################################
