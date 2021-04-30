import torch
import numpy as np
import random

from tqdm import tqdm

from betaincder import betainc, betaincderp, betaincderq
from torch import lgamma, digamma, log1p, exp, log

def deterministic(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def betaincderx(x, a, b):
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    partial_x = exp((b - 1) * log1p(-x) + (a - 1) * log(x) - lbeta)
    return partial_x

class BetaInc(torch.autograd.Function):
    """ regularized incomplete beta function and its forward and backward passes"""

    @staticmethod
    def forward(ctx, p, q, x):

        ctx.save_for_backward(p, q, x)
        # deal with dirac distributions
        if p == 0.:
            return torch.tensor(1) # for any x, cumulative = 1.

        elif q == 0. or x == 0.:
            return torch.tensor(0) # cumulative = 0.
    
        return torch.tensor(betainc(x, p, q))

    @staticmethod
    def backward(ctx, grad):
        p, q, x = ctx.saved_tensors

        if p == 0. or q == 0. or x == 0.: # deal with dirac distributions
            grad_p, grad_q, grad_x = 0., 0., 0.

        else:
            grad_p, grad_q, grad_x = betaincderp(x, p, q), betaincderq(x, p, q), betaincderx(x, p, q)

        return grad * grad_p, grad * grad_q, grad * grad_x