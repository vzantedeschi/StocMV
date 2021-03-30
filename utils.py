import jax.numpy as jnp

from jax import vmap, custom_vjp, grad

import cvxpy as cvx

def kl(q, p):
    return q * jnp.log(q/p) + (1-q) * jnp.log((1-q)/(1-p))

# @custom_vjp
# def kl_inv(q, err):
#     # bijection optimization
#     p_max, p_min = 1.0, q

#     for i in range(1000):
        
#         p = (p_min+p_max)/2.
#         p_kl = kl(q, p)

#         if jnp.isclose(p_kl, err) or (p_max-p_min)/2. < 1e-8:
#             return p

#         if p_kl > err:
#             p_max = p
#         else:
#             p_min = p

#     return p

@custom_vjp
def kl_inv(q, err):
    # QP optimization

    p = cvx.Variable(shape=1)

    prob = cvx.Problem(
        cvx.Maximize(p),
        [cvx.kl_div(q, p) + cvx.kl_div((1 - q),(1 - p)) <= err,
        1e-8 <= p, p <= 1-1e-8]
    )
    prob.solve()

    return p.value[0]

def kli_fwd(q, err):

    out = kl_inv(q, err)
    # out = jnp.clip(out, 1e-9, 1-1e-4)

    term_1 = (1.0 - q) / (1.0 - out)
    term_2 = q / out

    dev_q = jnp.log(term_1 / term_2) / (term_1 - term_2)
    dev_err = 1 / (term_1 - term_2)


    return out, (dev_q, dev_err)

def kli_bwd(res, g):
    
    dev_q, dev_err = res # Gets residuals computed in kli_fwd
    return (dev_q * g, dev_err * g)

kl_inv.defvjp(kli_fwd, kli_bwd)