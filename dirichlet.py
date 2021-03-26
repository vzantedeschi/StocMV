import jax.numpy as jnp
import numpy as np

from jax import vmap, custom_vjp, grad, jit
from jax.scipy.special import gammaln, digamma, gammainc
from jax.scipy.stats import gamma
from jax import random as jrand

from functools import partial

from betaincder import betainc, betaincderp, betaincderq

# regularized incomplete beta function and its forward and backward passes
@custom_vjp
def regbetainc(p, q, x):

    if p == 0.:
        return 1.
    elif q == 0.:
        return 0.
    else:
        return betainc(x, p, q)

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
def KL(alpha, beta):
    
    a = jnp.exp(alpha)
    b = jnp.exp(beta)

    res = gammaln(jnp.sum(a)) - jnp.sum(gammaln(a))
    res -= gammaln(jnp.sum(b)) - jnp.sum(gammaln(b))
    res += jnp.sum((a - b) * (digamma(a) - digamma(jnp.sum(a))))

    return res

# 01-loss applied to dataset
def risk(data, alpha):

    _, y_target, y_pred = data

    correct = jnp.where(y_pred == y_target, jnp.exp(alpha), 0.).sum(1)
    wrong = jnp.where(y_pred != y_target, jnp.exp(alpha), 0.).sum(1)

    s = [regbetainc(c, w, 0.5) for c, w in zip(correct, wrong)]
    return sum(s) / len(y_target)

def approximated_risk(data, alpha, loss, key):

    _, y_target, y_pred = data

    theta = jit(dirichlet_sampler)(jnp.exp(alpha), key)
    # import pdb; pdb.set_trace()
    return jit(loss)(y_target, y_pred, theta).mean()