import torch

from core.utils import BetaInc

def sigmoid_loss(y_target, y_pred, thetas, c=100):

    w_theta = torch.stack([torch.where(y_target != y_pred, t, torch.zeros(1)).sum(1) for t in thetas])

    return torch.sigmoid(c * (w_theta - 0.5))

def rand_loss(y_target, y_pred, theta, n=100):

    w_theta = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)

    return torch.stack([BetaInc.apply(torch.tensor(n // 2 + 1), torch.tensor(n // 2), w) for w in w_theta])

def moment_loss(y_target, y_pred, theta, order=1):

    assert order in [1, 2], "only first and second order supported atm"

    w_theta = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)

    return w_theta ** order

def exp_loss(y_target, y_pred, theta, c=1):

    w_theta = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)

    return torch.exp(c * (w_theta - 0.5)) / torch.tensor(c / 2).exp()
