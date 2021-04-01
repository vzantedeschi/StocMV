import torch

def sigmoid_loss(y_target, y_pred, thetas, c=100):

    w_theta = torch.stack([torch.where(y_target != y_pred, t, torch.zeros(1)).sum(1) for t in thetas])

    return torch.sigmoid(c * (w_theta - 0.5))

def moment_loss(y_target, y_pred, theta, order=1):

    assert order in [1, 2], "only first and second order supported atm"

    w_theta = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)

    return w_theta ** order