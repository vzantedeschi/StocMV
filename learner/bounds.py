import numpy as np

from core.kl_inv import klInvFunction

def mcallester_bound(data, posterior, cost, prior, delta, cost_params, coeff=1, verbose=False):

    n = len(data[0])
    
    kl = KL(posterior, prior)
    const = np.log(2 * (n**0.5) / delta)
    r = cost(data, posterior, *cost_params)

    bound = coeff * (r + ((kl + const) / 2 / n)**0.5)

    if verbose:
        print(f"Empirical risk={r}, KL={kl}, const={const}")
        print(f"Bound={bound}\n")

    return bound 

def seeger_bound(data, posterior, cost, prior, delta, cost_params, coeff=1, verbose=False):

    n = len(data[0])
    
    kl = KL(posterior, prior)
    const = np.log(2 * (n**0.5) / delta)
    r = cost(data, posterior, *cost_params)

    bound = coeff * kl_inv(r, (kl + const) / n)

    if verbose:
        print(f"Empirical risk={r}, KL={kl}, const={const}")
        print(f"Bound={bound}\n")

    return bound 

BOUNDS = {
    "mcallester": mcallester_bound,
    "seeger": seeger_bound
}