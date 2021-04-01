import numpy as np

from core.kl_inv import klInvFunction

def mcallester_bound(n, model, risk, delta, coeff=1, verbose=False):

    kl = model.KL()
    const = np.log(2 * (n**0.5) / delta)

    bound = coeff * (risk + ((kl + const) / 2 / n)**0.5)

    if verbose:
        print(f"Empirical risk={risk}, KL={kl}, const={const}")
        print(f"Bound={bound}\n")

    return bound 

def seeger_bound(n, model, risk, delta, coeff=1, verbose=False):
    
    kl = model.KL()
    const = np.log(2 * (n**0.5) / delta)

    bound = coeff * klInvFunction.apply(risk, (kl + const) / n)

    if verbose:
        print(f"Empirical risk={risk}, KL={kl}, const={const}")
        print(f"Bound={bound}\n")

    return bound 

BOUNDS = {
    "mcallester": mcallester_bound,
    "seeger": seeger_bound
}