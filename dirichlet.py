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
def error(params, sample, eps=1e-8):

    alpha, predictors = params
    x, y = sample

    y_pred = jnp.array([p(x) for p in predictors])

    correct = jnp.where(y_pred == y, alpha, 0.).sum()
    wrong = jnp.where(y_pred != y, alpha, 0.).sum()

    return regbetainc(correct+eps, wrong+eps, 0.5)

# empirical risk over the batch
def risk_vmap(alpha, predictors, batch):

    return jnp.mean(vmap(partial(error, (alpha, predictors)))(batch))

def risk_loop(alpha, predictors, batch):

    errs = jnp.array([error((alpha, predictors), (x, y)) for x, y in zip(*batch)])

    return jnp.mean(errs)
