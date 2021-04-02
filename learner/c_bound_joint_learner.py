#!/usr/bin/env python
import torch
import math

import cvxpy as cp

###############################################################################


class CBoundJointLearner():

    def __init__(
        self, delta=0.05, t=1e2
    ):
        self.t = t
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

    def __bound(self, kl, n, delta):
        # We compute the PAC-Bayes bound of PAC-Bound 2 (see page 820 of [1])
        b = math.log((2.* n**0.5 + n) / delta)
        b = (2.0 * kl + b) / n
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

    def __optimize_given_eS_dS(self, eS, dS, kl, n):
        # We solve the inner maximization Problem using the
        # "Bisection method for quasiconvex optimization" of [3] (p 146)
        u=1.0
        l=0.0
        bound = self.__bound(kl, n, self.delta).item()

        while(u-l>0.01):
            t = (l+u)/2.0

            e = cp.Variable(shape=1,nonneg=True)
            d = cp.Variable(shape=1,nonneg=True)

            prob = cp.Problem(
                cp.Minimize((1-(2*e+d))**2.0-t*(1-2*d)),
                [(cp.kl_div(eS,e)+cp.kl_div(dS,d)
                  +cp.kl_div((1-eS-dS), 1-e-d)<=bound),
                 2*e+d<=1,
                 d<=2.0*(cp.sqrt(e)-e),
                 d <= 0.5, e <= 0.5])

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

    def loss(self, n, model, batch):
        # We optimize the PAC-Bound 2 by gradient descent

        y_target, y_pred = batch
        mv_pred = torch.exp(model.post) * y_pred

        kl = model.KL()

        eS = torch.mean((0.5*(1.0-y_target*mv_pred))**2.0)
        dS = torch.mean(0.5*(1.0-(mv_pred**2.0)))

        if 2 * eS + dS >= 1:
            return 2 * eS + dS

        e, d = self.__optimize_given_eS_dS(eS.item(), dS.item(), kl, n)

        if e is not None and d is not None:
            e = torch.tensor(e, device=eS.device)
            d = torch.tensor(d, device=dS.device)

            b = self.__bound(kl, n, self.delta)

            loss = 1 - (1 - 2*e - d)**2 / (1 - 2 * d) 
            loss -= self.__log_barrier(self.__kl_tri(dS, eS, d, e) - b)
            loss -= self.__log_barrier(d - 2.* (min(e, 0.25) ** 0.5 - e))

            return loss

        return 1.

        # self._log["c-bound"] = self.__c_bound(e, d)
        # self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y_target*y_pred)))

###############################################################################

# References:
# [1] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
#     Pascal Germain, Alexandre Lacasse, Francois Laviolette, Mario Marchand, Jean-Francis Roy, 2015
# [2] Constrained Deep Networks: Lagrangian Optimization via Log-Barrier Extensions
#     Hoel Kervadec, Jose Dolz, Jing Yuan, Christian Desrosiers, Eric Granger, Ismail Ben Ayed, 2019
# [3] Convex Optimization
#     Stephen Boyd, Lieven Vandenberghe, 2004
