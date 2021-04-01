import torch

def sigmoid_loss(y_target, y_pred, theta, c=10):

    w_theta = theta[y_pred != y_target].sum(1)

    return torch.sigmoid(c * (w_theta - 0.5))

def moment_loss(y_target, y_pred, theta, order=1):

    assert order in [1, 2], "only first and second order supported atm"

    w_theta = theta[y_pred != y_target].sum(1)

    return w_theta ** order