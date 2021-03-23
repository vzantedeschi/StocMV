#!/usr/bin/env python
import torch
import math
from learner.gradient_descent_learner import GradientDescentLearner
import cvxpy as cp
from core.cocob_optim import COCOB

###############################################################################


class CBoundJointLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, batch_size=None, delta=0.05, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.t = t
        self._optim = None
        self.delta = delta

    def __c_bound(self, e, d):
        # We compute the C-Bound of PAC-Bound 2 (see page 820 of [1])
        return (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))

    def __log_barrier(self, x):
        assert isinstance(x, torch.Tensor) and len(x.shape) == 0
        # We use the log-barrier extension of [2]
        if(x <= -1.0/(self.t**2.0)):
            return -(1.0/self.t)*torch.log(-x)
        else:
            return self.t*x - (1.0/self.t)*math.log(1/(self.t**2.0))+(1/self.t)

    def __bound(self, kl, m, delta):
        # We compute the PAC-Bayes bound of PAC-Bound 2 (see page 820 of [1])
        b = math.log((2.0*math.sqrt(m)+m)/delta)
        b = (1.0/m)*(2.0*kl+b)
        return b

    def __kl_tri(self, q1, q2, p1, p2):
        # We compute the KL divergence between two trinomials
        # (see eq. (31) of [1])
        kl = torch.tensor(0.0).to(q1.device)
        if(q1 > 0):
            kl += q1*torch.log(q1/p1)
        if(q2 > 0):
            kl += q2*torch.log(q2/p2)
        if(q1+q2 < 1):
            kl += (1-q1-q2)*torch.log((1-q1-q2)/(1-p1-p2))
        return kl

    def __optimize_given_eS_dS(self, eS, dS, kl, m):
        # We solve the inner maximization Problem using the
        # "Bisection method for quasiconvex optimization" of [3] (p 146)
        u=1.0
        l=0.0
        bound = self.__bound(kl, m, self.delta).item()

        while(u-l>0.01):
            t = (l+u)/2.0

            e = cp.Variable(shape=1,nonneg=True)
            d = cp.Variable(shape=1,nonneg=True)

            prob = cp.Problem(
                cp.Minimize((1-(2*e+d))**2.0-t*(1-2*d)),
                [(cp.kl_div(eS,e)+cp.kl_div(dS,d)
                  +cp.kl_div((1-eS-dS), 1-e-d)<=bound),
                 2*e+d<=1,
                 d<=2.0*(cp.sqrt(e)-e)])

            prob.solve()

            if(e.value is None or d.value is None):
                # Only in case where the solution is not found
                return (None, None)
            else:
                e=e.value[0]
                d=d.value[0]

            c_bound = 1.0-((1-(2*e+d))**2.0)/(1-2*d)

            if(c_bound > 1.0-t):
                u = t
            else:
                l = t
        return (e, d)

    def __optimize_given_e_d(self, e, d, eS, dS, kl, m):
        # We compute the gradient descent step given (e,d)
        e = torch.tensor(e, device=eS.device)
        d = torch.tensor(d, device=dS.device)

        b = self.__bound(kl, m, self.delta)
        self._loss = -self.__log_barrier(self.__kl_tri(dS, eS, d, e)-b)
        self._loss += -self.__log_barrier(-0.5*(2.0*eS+dS))

        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()


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
        dS = torch.mean(0.5*(1.0-(pred**2.0)))
        m = batch["m"]

        (e, d) = self.__optimize_given_eS_dS(
            eS.item(), dS.item(), kl, m)
        if(e is not None and d is not None):
            self.__optimize_given_e_d(e, d, eS, dS, kl, m)

        self._log["c-bound"] = self.__c_bound(e, d)
        self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y*pred)))

###############################################################################

# References:
# [1] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
#     Pascal Germain, Alexandre Lacasse, Francois Laviolette, Mario Marchand, Jean-Francis Roy, 2015
# [2] Constrained Deep Networks: Lagrangian Optimization via Log-Barrier Extensions
#     Hoel Kervadec, Jose Dolz, Jing Yuan, Christian Desrosiers, Eric Granger, Ismail Ben Ayed, 2019
# [3] Convex Optimization
#     Stephen Boyd, Lieven Vandenberghe, 2004
