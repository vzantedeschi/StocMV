import torch
import numpy as np

from torch.distributions.dirichlet import Dirichlet
from torch import lgamma, digamma

from betaincder import betainc, betaincderp, betaincderq

class BetaInc(torch.autograd.Function):
    """ regularized incomplete beta function and its forward and backward passes"""

    @staticmethod
    def forward(ctx, p, q, x):

        ctx.save_for_backward(p, q, torch.tensor(x))
        # deal with dirac distributions
        if p == 0.:
            return torch.ones(1) # for any x, cumulative = 1.

        elif q == 0.:
            return torch.zeros(1) # for any x > 0, cumulative = 0.
    
        return torch.tensor(betainc(x, p, q))

    @staticmethod
    def backward(ctx, grad):
        p, q, x = ctx.saved_tensors

        grad_p, grad_q = betaincderp(x, p, q), betaincderq(x, p, q)

        return grad * grad_p, grad * grad_q, None

class DirichletCustom():

    def __init__(self, alpha):

        self.alpha = alpha

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        exp_alpha = torch.exp(self.alpha)
        res = lgamma(exp_alpha.sum()) - lgamma(exp_alpha).sum()
        res -= lgamma(beta.sum()) - lgamma(beta).sum()
        res += torch.sum((exp_alpha - beta) * (digamma(exp_alpha) - digamma(exp_alpha.sum())))

        return res

    def risk(self, batch):
        # 01-loss applied to batch

        y_target, y_pred = batch
        exp_alpha = torch.exp(self.alpha)

        correct = torch.where(y_target == y_pred, exp_alpha, torch.zeros(1)).sum(1)
        wrong = torch.where(y_target != y_pred, exp_alpha, torch.zeros(1)).sum(1)

        s = [BetaInc.apply(c, w, 0.5) for c, w in zip(correct, wrong)]

        return sum(s) / len(y_target)

    def approximated_risk(self, batch, loss, num_draws=10):

        y_target, y_pred = batch

        thetas = Dirichlet(torch.exp(self.alpha)).rsample((num_draws,))

        return loss(y_target, y_pred, thetas).mean()

class Categorical():

    def __init__(self, theta):
        self.theta = theta

    def KL(self, beta):

        exp_theta = torch.exp(self.theta)
        t = exp_theta / exp_theta.sum()

        b = beta / beta.sum()

        return (t * torch.log(t / b)).sum()

    def approximated_risk(self, batch, loss):

        exp_theta = torch.exp(self.theta)
        t = exp_theta / exp_theta.sum()

        y_target, y_pred = batch

        return loss(y_target, y_pred, t).mean()

    def risk(self, batch):

        exp_theta = torch.exp(self.theta)
        t = exp_theta / exp_theta.sum()

        y_target, y_pred = batch

        w_theta = torch.where(y_target != y_pred, t, torch.zeros(1)).sum(1)

        return (w_theta >= 0.5).float().mean()

distr_dict = {
    "dirichlet": DirichletCustom,
    "categorical": Categorical
}