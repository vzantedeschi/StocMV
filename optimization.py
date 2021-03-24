import jax.numpy as jnp

from jax import grad, jit
from jax.experimental.optimizers import adam

def batch_gradient_descent(cost, alpha, params, lr=0.1, num_iters=1000):

    grad_alpha = grad(cost, argnums=0)

    for i in range(num_iters):
        
        g = grad_alpha(alpha, *params)
        alpha -= lr * g
        alpha = jnp.clip(alpha, 0)

        if i % 100 == 0:
            print(f"alpha={alpha}, objective={cost(alpha, *params)}")

    return alpha

# def jax_(cost, alpha, params, lr=0.1, b1=0.9, b2):

#     opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
#     opt_state = opt_init(params)

#     def step(step, opt_state):
#       value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
#       opt_state = opt_update(step, grads, opt_state)
#       return value, opt_state

#     for i in range(num_steps):
#       value, opt_state = step(i, opt_state)

#     grad_alpha = grad(cost, argnums=1)

#     for i in range(num_iters):

#         alpha += lr * grad_alpha(alpha, *params)

#         if i % 10 == 0:
#             print(alpha, cost(alpha, *params))

#     return alpha