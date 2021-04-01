import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from voter.majority_vote import MajorityVote

###############################################################################

class DecisionStump(BaseEstimator, ClassifierMixin):

    def __init__(self, feature, threshold, dir):
        self.feature = feature
        self.threshold = threshold
        self.dir = dir

        self.fit()

    def fit(self):
        return self

    def output(self, X):
        # X -> (size, nb_feature)
        assert ((isinstance(X, torch.Tensor) or isinstance(X, np.ndarray))
                and (len(X.shape)== 2))

        if(isinstance(X, torch.Tensor)):
            out = X[:, self.feature].unsqueeze(1)
            out = self.dir*(2.0*(out > self.threshold).float()-1.0)
        else:
            out = np.expand_dims(X[:, self.feature], 1)
            out = self.dir*(2.0*(out > self.threshold).astype(float)-1.0)
        return out


class DecisionStumpMV(MajorityVote):

    def __init__(
        self, X, y,
        nb_per_attribute=10, complemented=False, quasi_uniform=False):
        self.nb_per_attribute = nb_per_attribute

        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(X.shape) == 2 and len(y.shape) == 2 and
                X.shape[0] == y.shape[0] and
                y.shape[1] == 1 and X.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        super().__init__(
            X, y, complemented=complemented, quasi_uniform=quasi_uniform)

        self.fit()

    def fit(self):
        # X -> (size, nb_feature)
        # y -> (size, 1)
        super().fit()
        X, y = self.X_y_list
        x_min_list = np.min(X, axis=0)
        x_max_list = np.max(X, axis=0)
        gap_list = (x_max_list-x_min_list)/(self.nb_per_attribute + 1)

        dir_list = [+1]
        if(self.complemented):
            dir_list = [+1, -1]

        for dir in dir_list:
            for i in range(len(X[0])):
                gap = gap_list[i]
                x_min = x_min_list[i]

                if gap == 0:
                    continue
                for t in range(self.nb_per_attribute):
                    self.voter_list.append(
                        DecisionStump(i, x_min+gap*(t+1), dir))

        if(not(self.complemented) and self.quasi_uniform):
            self.prior = np.zeros((len(self.voter_list), 1))
        else:
            self.prior = ((1.0/len(self.voter_list))
                          *np.ones((len(self.voter_list), 1)))

        self.post = np.array(self.prior)

        return self
