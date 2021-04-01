import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from voter.stump import DecisionStumpMV


class Learner(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super(Learner, self).__init__()

    def fit(self, X, y):
        raise NotImplementedError()