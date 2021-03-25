import jax.numpy as jnp

from dirichlet import *

def mcallester_bound(data, alpha, cost, beta, delta, params, verbose=False):

    n = len(data[0])
    
    kl = KL(alpha, beta)
    const = jnp.log(2 * (n**0.5) / delta)

    r = cost(data, alpha, *params) # compute risk of 01-loss

    bound = r + ((kl + const) / 2 / n)**0.5

    if verbose:
        print(f"Empirical risk={r}, KL={kl}, const={const}")
        print(f"Bound={bound}\n")

    return bound 
