#!/usr/bin/env python
import torch
import math
import copy
from learner.gradient_descent_learner import GradientDescentLearner
import numpy as np
from core.cocob_optim import COCOB
from core.kl_inv import klInvFunction


###############################################################################
class BoundJointLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, batch_size=None, delta=0.05, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.t = t
        self._optim = None
        self.delta = delta

    def __bound(self, kl, m):
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/self.delta))

    def __joint_bound(self, eS, kl, m):
        kl_inv = klInvFunction.apply
        b = self.__bound(2.0*kl, m)
        b = kl_inv(eS, b, "MAX")
        return b

    def _optimize(self, batch):
        # We optimize the PAC-Bound 2 by gradient descent
        if(self._optim is None):
            self._optim = COCOB(self.mv_diff.parameters())

        self.mv_diff(batch)
        pred = self.mv_diff.pred
        kl = self.mv_diff.kl

        assert "y" in batch and isinstance(batch["y"], torch.Tensor)
        y = batch["y"]
        y_unique = torch.sort(torch.unique(y))[0]
        assert y_unique[0].item() == -1 and y_unique[1].item() == +1

        assert len(y.shape) == 2 and len(pred.shape) == 2
        assert (pred.shape[0] == y.shape[0] and pred.shape[1] == y.shape[1]
                and y.shape[1] == 1)
        assert len(kl.shape) == 0

        eS = torch.mean((0.5*(1.0-y*pred))**2.0)
        m = batch["m"]

        e = self.__joint_bound(eS, kl, m)
        self._loss = 4.0*e

        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

        self._log["bound"] = self._loss
        self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y*pred)))

###############################################################################
