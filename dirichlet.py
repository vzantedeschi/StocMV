import jax.numpy as jnp
import numpy as np

from jax import vmap, custom_vjp, grad
from jax.scipy.special import gammaln, digamma, gammainc
from jax.scipy.stats import gamma
from jax import random as jrand

from functools import partial

from betaincder import betainc, betaincderp, betaincderq

# regularized incomplete beta function and its forward and backward passes
@custom_vjp
def regbetainc(p, q, x):
    b = betainc(x, p, q)
    return b

def b_fwd(p, q, x):

    return betainc(x, p, q), (betaincderp(x, p, q), betaincderq(x, p, q))

def b_bwd(res, g):
    
    dev_p, dev_q = res # Gets residuals computed in b_fwd
    return (dev_p * g, dev_q * g, None)

regbetainc.defvjp(b_fwd, b_bwd)

def dirichlet_sampler(alpha, key):

    theta = vmap(jrand.gamma, (None, 0))(key, alpha) # draw from Gamma
    theta /= sum(theta) # now theta is drawn from Dirichlet

    return theta

# Kullback-Leibler divergence between two Dirichlets
def KL(alpha, beta, eps=1e-8):
    
    res = gammaln(jnp.sum(alpha) + eps) - jnp.sum(gammaln(alpha + eps))
    res -= gammaln(jnp.sum(beta) + eps) - jnp.sum(gammaln(beta + eps))
    res += jnp.sum((alpha - beta) * (digamma(alpha + eps) - digamma(jnp.sum(alpha) + eps)))

    return res

# 01-loss applied to dataset
def risk(alpha, predictors, sample, loss=None, eps=1e-8):

    x, y = sample
    y_target = y[..., None]

    y_pred = predictors(x)
    # import pdb; pdb.set_trace()

    correct = jnp.where(y_pred == y_target, alpha, 0.).sum(1)
    wrong = jnp.where(y_pred != y_target, alpha, 0.).sum(1)

    return sum([regbetainc(c+eps, w+eps, 0.5) for c, w in zip(correct, wrong)]) / len(x)

def approximated_risk(alpha, predictors, sample, loss, key, eps=1e-8):
    
    x, y = sample
    y_target = y[..., None]

    y_pred = predictors(x)

    theta = dirichlet_sampler(alpha, key)

    return loss(y_target, y_pred, theta).mean()