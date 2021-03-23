import jax.numpy as jnp
import numpy as np

from jax import vmap, custom_vjp
from jax.scipy.special import gammaln, digamma

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
    return (dev_p * g, dev_q * g, 0)

regbetainc.defvjp(b_fwd, b_bwd)

# Kullback-Leibler divergence between two Dirichlets
def KL(alpha, beta):

    res = gammaln(jnp.sum(alpha)) - jnp.sum(gammaln(alpha))
    res -= gammaln(jnp.sum(beta)) - jnp.sum(gammaln(beta))
    res += jnp.sum((alpha - beta) * (digamma(alpha) - digamma(jnp.sum(alpha))))

    return res

# 01-loss for one point
def risk(alpha, predictors, sample, eps=1e-8):

    x, y = sample

    y_pred = predictors(x)
    # import pdb; pdb.set_trace()

    correct = jnp.where(y_pred == y[:, None], alpha, 0.).sum(1)
    wrong = jnp.where(y_pred != y[:, None]  , alpha, 0.).sum(1)

    return sum([regbetainc(c+eps, w+eps, 0.5) for c, w in zip(correct, wrong)]) / len(x)
