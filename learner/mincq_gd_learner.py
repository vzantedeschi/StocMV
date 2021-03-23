import torch
import math
from learner.gradient_descent_learner import GradientDescentLearner
from core.cocob_optim import COCOB

###############################################################################

class MinCqGDLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, mu, epoch=1, batch_size=None, t=100
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self.mu = mu
        self.t = t
        self._optim = None

    def __c_bound(self, mu_1, mu_2):
        # We compute the "Second Form" of the C-Bound (See [1])
        mu_1 = torch.max(torch.tensor(0.0).to(mu_1.device), mu_1)
        mu_2 = torch.min(torch.tensor(1.0).to(mu_2.device), mu_2)
        cb = (1.0-(mu_1**2.0)/(mu_2))
        if(torch.isnan(cb) or torch.isinf(cb)):
            cb = 1.0
        return cb

    def __log_barrier(self, x):
        assert isinstance(x, torch.Tensor) and len(x.shape)==0
        # We use the log-barrier extension of [2]
        if(x <= -1.0/(self.t**2.0)):
            return -(1.0/self.t)*torch.log(-x)
        else:
            return self.t*x - (1.0/self.t)*math.log(1/(self.t**2.0))+(1/self.t)

    def _optimize(self, batch):
        if(self._optim is None):
            self._optim = COCOB(self.mv_diff.parameters())

        self.mv_diff(batch)
        pred = self.mv_diff.pred

        assert "y" in batch and isinstance(batch["y"], torch.Tensor)
        y = batch["y"]
        y_unique = torch.sort(torch.unique(y))[0]
        assert y_unique[0].item() == -1 and y_unique[1].item() == +1

        assert len(y.shape) == 2 and len(pred.shape) == 2
        assert (pred.shape[0] == y.shape[0] and pred.shape[1] == y.shape[1]
                and y.shape[1] == 1)

        mu_1 = torch.mean(y*pred)
        mu_2 = torch.mean(pred**2.0)

        cb = self.__c_bound(mu_1, mu_2)
        self._loss = cb - self.__log_barrier(-(torch.abs(self.mu-mu_1)-0.001))

        self._log["mu_1"] = mu_1
        self._log["mu"] = self.mu
        self._log["c-bound"] = cb
        self._log["0-1 loss"] = 0.5*(1.0-torch.mean(torch.sign(y*pred)))

        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

###############################################################################

# References:
# [1] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
#     Pascal Germain, Alexandre Lacasse, Francois Laviolette, Mario Marchand, Jean-Francis Roy, 2015
# [2] Constrained Deep Networks: Lagrangian Optimization via Log-Barrier Extensions
#     Hoel Kervadec, Jose Dolz, Jing Yuan, Christian Desrosiers, Eric Granger, Ismail Ben Ayed, 2019
