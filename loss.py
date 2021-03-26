import jax.numpy as jnp
from jax.nn import sigmoid

def sigmoid_loss(y_target, y_pred, theta, c=10):

    wrong = jnp.where(y_pred != y_target, theta, 0.).sum(1)

    return sigmoid(c * (wrong - 0.5))

def moment_loss(y_target, y_pred, theta, order=1):

    assert order in [1, 2], "only first and second order supported"

    w_theta = jnp.where(y_pred != y_target, theta, 0.).sum(1)

    return (w_theta ** order).mean()