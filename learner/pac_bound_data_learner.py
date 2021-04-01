#!/usr/bin/env python
import sys
import argparse
import random
import torch
import math
import numpy as np
import logging
import copy

from sklearn.base import BaseEstimator, ClassifierMixin
from models.models import Module
from core.numpy_dataset import NumpyDataset


###############################################################################
class PACBoundDataLearner(BaseEstimator, ClassifierMixin):

    def __create_bound(self, m1, m2):

        def bound(kl, m, delta):
            assert m == m1+m2
            # Bound of the "PAC-Bound 2"
            b = math.log((2.0*math.sqrt(m1)+m1)/delta)
            b = b+math.log((2.0*math.sqrt(m2)+m2)/delta)
            b = (1.0/m)*(2.0*kl+b)
            #  b = (1.0/m2)*(2.0*kl+b)
            return b
        return bound

    def __init__(self, learner_prior, learner_post):
        self.learner_1 = copy.deepcopy(learner_prior)
        self.learner_2 = copy.deepcopy(self.learner_1)
        self.learner = copy.deepcopy(learner_post)
        self.learner.lr_test = 0.00005

    def fit(self, X, y):

        SEED = 0

        X_1 = X[X.shape[0]//2:]
        y_1 = y[y.shape[0]//2:]
        X_2 = X[:X.shape[0]//2]
        y_2 = y[:y.shape[0]//2]

        bound = self.__create_bound(X_1.shape[0], X_2.shape[0])
        self.learner.bound = bound
        #  print(self.learner.bound(0, 100, 0.05))

        #  raise Exception("OK")

        logging.info("Running the training [1/3]\n")
        self.learner_1.fit(X_1, y_1)
        save_1 = self.learner_1.save()["post"]
        save_1 = torch.abs(save_1)/torch.sum(torch.abs(save_1))
        logging.info("Running the training [2/3]\n")
        self.learner_2.fit(X_2, y_2)
        save_2 = self.learner_2.save()["post"]
        save_2 = torch.abs(save_2)/torch.sum(torch.abs(save_2))
        logging.info("Running the training [3/3]\n")
        save = 0.5*(save_1+save_2)
        save = save/torch.sum(save)
        #  save = save_1
        self.learner.model.prior_1 = copy.deepcopy(save_1)
        self.learner.model.prior_2 = copy.deepcopy(save_1)
        #  self.learner.model.prior = copy.deepcopy(save)
        self.learner.model.post = torch.nn.Parameter(copy.deepcopy(save))
        self.learner.fit(X, y)

        return self.learner

    def output(self, X):
        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        output = None
        for i, batch in enumerate(loader):
            self.model(batch)
            if(output is None):
                output = self.model.output.detach().numpy()
            else:
                output = np.concatenate(
                    (output, self.model.output.detach().numpy()))
        return output[:, 0]

    def predict(self, X):
        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        pred = None
        for i, batch in enumerate(loader):
            self.model(batch)
            if(pred is None):
                pred = self.model.predict.detach().numpy()
            else:
                pred = np.concatenate(
                    (pred, self.model.predict.detach().numpy()))
        return pred[:, 0]

    def predict_proba(self, X):
        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)
        pred = None
        for i, batch in enumerate(loader):
            self.model(batch)
            if(pred is None):
                pred = self.model.predict_proba.detach().numpy()
            else:
                pred = np.concatenate(
                    (pred, self.model.predict_proba.detach().numpy()))
        return pred

    def divergence(self):
        return self.model.kl.detach().numpy()

    def save(self):
        return self.model.state_dict()

    def load(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def _optimize(self):
        raise NotImplementedError


###############################################################################
