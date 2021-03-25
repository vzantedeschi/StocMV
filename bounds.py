import jax.numpy as jnp

from dirichlet import *

def mcallester_bound(alpha, beta, delta, predictors, data, loss=None, key=None, verbose=False):

    n = len(data[0])
    
    kl = KL(alpha, beta)
    const = jnp.log(2 * (n**0.5) / delta)

    if loss is None:
        r = risk(alpha, predictors, data) # compute risk of 01-loss

    else: # compute risk for the given loss
        r = approximated_risk(alpha, predictors, data, loss=loss, key=key)

    bound = r + ((kl + const) / 2 / n)**0.5

    if verbose:
        print(f"Empirical risk={r}, KL={kl}, const={const}")
        print(f"Bound={bound}")

    return bound 