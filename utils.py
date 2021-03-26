import jax.numpy as jnp

from jax import vmap, custom_vjp, grad

import cvxpy as cvx

@custom_vjp
def kl_inv(q, err):

    p = cvx.Variable(shape=1)

    prob = cvx.Problem(
        cvx.Maximize(p),
        [cvx.kl_div(q, p) + cvx.kl_div((1 - q),(1 - p)) <= err])
    prob.solve()

    return p.value[0]

def kli_fwd(q, err):

    out = kl_inv(q, err)
    out = jnp.clip(out, 1e-9)

    term_1 = (1.0 - q) / (1.0 - out)
    term_2 = q / out

    dev_q = jnp.log(term_1 / term_2) / (term_1 - term_2)
    dev_err = 1 / (term_1 - term_2)

    return out, (dev_q, dev_err)

def kli_bwd(res, g):
    
    dev_q, dev_err = res # Gets residuals computed in kli_fwd
    return (dev_q * g, dev_err * g)

kl_inv.defvjp(kli_fwd, kli_bwd)