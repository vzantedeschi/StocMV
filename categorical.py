import jax.numpy as jnp

from jax import jit

def KL(alpha, beta):

    a = jnp.exp(alpha)
    a /= a.sum()
    
    b = jnp.exp(beta)
    b /= b.sum()

    return (a * jnp.log(a / b)).sum()

def risk_upper_bound(data, theta, loss):

    t = jnp.exp(theta) / jnp.exp(theta).sum()

    _, y_target, y_pred = data

    return jit(loss)(y_target, y_pred, t).mean()

def risk(data, theta):

    t = jnp.exp(theta) / jnp.exp(theta).sum()

    _, y_target, y_pred = data

    w_theta = jnp.where(y_pred != y_target, t, 0.).sum(1)

    return (w_theta >= 0.5).sum() / len(y_pred)