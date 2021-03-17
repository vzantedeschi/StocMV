import jax.numpy as jnp
import numpy as np

from jax import vmap
from jax.scipy.special import gammaln, digamma, betainc

from functools import partial

def KL(alpha, beta):

    res = gammaln(jnp.sum(alpha)) - jnp.sum(gammaln(alpha))
    res -= gammaln(jnp.sum(beta)) - jnp.sum(gammaln(beta))
    res += jnp.sum((alpha - beta) * (digamma(alpha) - digamma(jnp.sum(alpha))))

    return res

def error(params, sample, eps=1e-8):

    alpha, predictors = params
    x, y = sample

    y_pred = jnp.array([p(x) for p in predictors])

    # overcomplicated solution to avoid errors due to boolean indexing in jax
    correct = jnp.where(y_pred == y, np.arange(len(alpha)), len(alpha))
    wrong = jnp.where(y_pred != y, np.arange(len(alpha)), len(alpha))

    wrong = alpha[wrong].sum()
    correct = alpha[correct].sum()

    return betainc(correct+eps, wrong+eps, 0.5)

def risk(alpha, predictors, batch):

    return jnp.mean(vmap(partial(error, (alpha, predictors)))(batch))
