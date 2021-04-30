import torch
import numpy as np

from torch.distributions.dirichlet import Dirichlet
from torch import lgamma, digamma

from core.utils import BetaInc

class DirichletCustom():

    def __init__(self, alpha, mc_draws=10):

        self.alpha = alpha
        self.mc_draws = mc_draws

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        exp_alpha = torch.exp(self.alpha)
        res = lgamma(exp_alpha.sum()) - lgamma(exp_alpha).sum()
        res -= lgamma(beta.sum()) - lgamma(beta).sum()
        res += torch.sum((exp_alpha - beta) * (digamma(exp_alpha) - digamma(exp_alpha.sum())))

        return res

    def risk(self, batch, mean=True):
        # 01-loss applied to batch
        y_target, y_pred = batch
        exp_alpha = torch.exp(self.alpha)

        correct = torch.where(y_target == y_pred, exp_alpha, torch.zeros(1)).sum(1)
        wrong = torch.where(y_target != y_pred, exp_alpha, torch.zeros(1)).sum(1)
        
        s = [BetaInc.apply(c, w, torch.tensor(0.5)) for c, w in zip(correct, wrong)]

        if mean:
            return sum(s) / len(y_target)

        return sum(s)

    def approximated_risk(self, batch, loss, mean=True):

        y_target, y_pred = batch

        thetas = self.rsample()

        r = loss(y_target, y_pred, thetas)

        if mean:
            return r.mean()

        return r.sum(0)

    def rsample(self):

        return Dirichlet(torch.exp(self.alpha)).rsample((self.mc_draws,))

class Categorical():

    def __init__(self, theta, mc_draws=10):
        self.theta = theta
        self.mc_draws = mc_draws
        
    def KL(self, beta):

        t = self.get_theta()

        b = beta / beta.sum()

        return (t * torch.log(t / b)).sum()

    def approximated_risk(self, batch, loss, mean=True):

        t = self.get_theta()

        y_target, y_pred = batch

        r = loss(y_target, y_pred, t)
        
        if mean:
            return r.mean()

        return r.sum()

    def risk(self, batch, mean=True):

        t = self.get_theta()

        y_target, y_pred = batch

        w_theta = torch.where(y_target != y_pred, t, torch.zeros(1)).sum(1)

        r = (w_theta >= 0.5).float()

        if mean:
            return r.mean()

        return r.sum()

    def rsample(self):

        t = self.get_theta()

        return t.unsqueeze(0)

    def get_theta(self):
        
        return torch.nn.functional.softmax(self.theta, dim=0)

distr_dict = {
    "dirichlet": DirichletCustom,
    "categorical": Categorical
}
