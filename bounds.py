import jax.numpy as jnp

from dirichlet import *
from utils import kl_inv

def mcallester_bound(data, alpha, cost, beta, delta, params, coeff=1, verbose=False):

    n = len(data[0])
    
    kl = KL(alpha, beta)
    const = jnp.log(2 * (n**0.5) / delta)
    r = cost(data, alpha, *params)

    bound = r + ((kl + const) / 2 / n)**0.5

    if verbose:
        print(f"Empirical risk={r}, KL={kl}, const={const}")
        print(f"Bound={bound}\n")

    return coeff * bound 

def seeger_bound(data, alpha, cost, beta, delta, params, coeff=1, verbose=False):

    n = len(data[0])
    
    kl = KL(alpha, beta)
    const = jnp.log(2 * (n**0.5) / delta)
    r = cost(data, alpha, *params)

    bound = kl_inv(r, (kl + const) / n)

    if verbose:
        print(f"Empirical risk={r}, KL={kl}, const={const}")
        print(f"Bound={bound}\n")

    return coeff * bound 

BOUNDS = {
    "mcallester": mcallester_bound,
    "seeger": seeger_bound
}