import jax.numpy as jnp
import numpy as np

from jax import vmap, custom_vjp
from jax.scipy.special import gammaln, digamma
from jax.scipy.stats import beta

from functools import partial

# regularized incomplete beta function and its forward and backward passes
def regbetainc(p, q, x, npts=100):
    
    t = jnp.linspace(1e-8, x, npts)
    
    return jnp.trapz(jnp.exp(beta.logpdf(t, p, q)), t)

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

# 01-loss for one point
def risk(alpha, predictors, sample, eps=1e-8):

    x, y = sample

    y_pred = jnp.stack([p(x) for p in predictors], 1)

    # import pdb; pdb.set_trace()
    correct = jnp.where(y_pred == y[:, None], alpha, 0.).sum(1)
    wrong = jnp.where(y_pred != y[:, None]  , alpha, 0.).sum(1)

    return sum([regbetainc(c+eps, w+eps, 0.5) for c, w in zip(correct, wrong)]) / len(x)
