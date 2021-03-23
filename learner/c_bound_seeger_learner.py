#!/usr/bin/env python
import torch
import math
import copy
from learner.gradient_descent_learner import GradientDescentLearner
import numpy as np
import cvxpy as cp
from core.cocob_optim import COCOB
from core.kl_inv import klInvFunction


###############################################################################
class CBoundSeegerLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, batch_size=None, delta=0.05, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.t = t
        self._optim = None
        self.delta = delta

    def __c_bound(self, r, d):
        # C-Bound
        r = torch.min(torch.tensor(0.5).to(r.device), r)
        d = torch.max(torch.tensor(0.0).to(d.device), d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        if(torch.isnan(cb) or torch.isinf(cb)):
            cb = torch.tensor(1.0, requires_grad=True)
        return cb

    def __bound(self, kl, m):
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/(0.5*self.delta)))

    def __risk_bound(self, rS, kl, m):
        kl_inv = klInvFunction.apply
        b = self.__bound(kl, m)
        b = kl_inv(rS, b, "MAX")
        return b

    def __disagreement_bound(self, dS, kl, m):
        kl_inv = klInvFunction.apply
        b = self.__bound(2.0*kl, m)
        b = kl_inv(dS, b, "MIN")
        return b

    def __log_barrier(self, x):
        assert isinstance(x, torch.Tensor) and len(x.shape)==0
        # We use the log-barrier extension of [2]
        if(x <= -1.0/(self.t**2.0)):
            return -(1.0/self.t)*torch.log(-x)
        else:
            return self.t*x - (1.0/self.t)*math.log(1/(self.t**2.0))+(1/self.t)

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

        rS = torch.mean((0.5*(1.0-y*pred)))
        dS = torch.mean(0.5*(1.0-(pred**2.0)))
        m = batch["m"]

        r = self.__risk_bound(rS, kl, m)
        d = self.__disagreement_bound(dS, kl, m)

        #if(rD >= 0.5):
        #    self._loss = 2.0*rD
        #else:
        self._loss = self.__c_bound(r, d)
        self._loss += -self.__log_barrier(-r)

        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

        self._log["c-bound"] = self.__c_bound(r, d)
        self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y*pred)))

###############################################################################
