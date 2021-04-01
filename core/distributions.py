import torch
import numpy as np

from torch.distributions.dirichlet import Dirichlet
from torch.special import gammaln, digamma

from betaincder import betainc, betaincderp, betaincderq

class BetaInc(torch.autograd.Function):
    """ regularized incomplete beta function and its forward and backward passes"""

    @staticmethod
    def forward(ctx, p, q, x):

        ctx.save_for_backward(p, q, x)

        # deal with dirac distributions
        if p == 0.:
            return 1. # for any x, cumulative = 1.

        elif q == 0.:
            return 0. # for any x > 0, cumulative = 0.
    
        return betainc(x, p, q)

    @staticmethod
    def backward(ctx, grad):
        p, q, x = ctx.saved_tensors

        grad_p, grad_q = betaincderp(x, p, q), betaincderq(x, p, q)

        return grad * grad_p, grad * grad_q, None

class DirichletCustom(Dirichlet):

    def __init__(self, alpha):

        super(DirichletCustom, self).__init__(concentration=alpha)

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        res = gammaln(self.concentration.sum()) - gammaln(self.concentration).sum()
        res -= gammaln(beta.sum()) - gammaln(beta).sum()
        res += torch.sum((self.concentration - beta) * (digamma(self.concentration) - digamma(self.concentration.sum())))

        return res

    def risk(self, batch):
        # 01-loss applied to batch

        _, y_target, y_pred = batch

        correct = self.concentration[y_target == y_pred].sum(1)
        wrong = self.concentration[y_target != y_pred].sum(1)

        s = torch.vmap(regbetainc)(correct, wrong, 0.5)

        return s.mean()

    def approximated_risk(self, batch, loss, num_draws=10):

        _, y_target, y_pred = batch

        thetas = self.rsample(num_draws)

        return loss(y_target, y_pred, theta).mean()

class Categorical():

    def __init__(self, theta):
        self.theta = theta

    def KL(self, beta):

        t = self.theta / self.theta.sum()
        b = beta / beta.sum()

        return (t * torch.log(t / b)).sum()

    def approximated_risk(self, batch, loss):

        t = self.theta / self.theta.sum()

        _, y_target, y_pred = batch

        return loss(y_target, y_pred, t).mean()

    def risk(self, batch):

        t = self.theta / self.theta.sum()

        _, y_target, y_pred = batch

        w_theta = t[y_pred != y_target].sum(1)

        return (w_theta >= 0.5).mean()

distr_dict = {
    "dirichlet": DirichletCustom,
    "categorical": Categorical
}