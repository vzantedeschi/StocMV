import numpy as np
import random

import jax.numpy as jnp
import jax.random as jrand

from datasets import *
from dirichlet import *
from optimization import batch_gradient_descent
from predictors import uniform_decision_stumps, custom_decision_stumps

def mcallester_bound(alpha, beta, delta, predictors, data, verbose=False):

    n = len(data[0])
    
    kl = KL(alpha, beta)
    tr = risk_loop(alpha, predictors, data)
    const = jnp.log(2 * (n**0.5) / delta)

    bound = tr + ((kl + const) / 2 / n)**0.5

    if verbose:
        print(f"True risk={tr}, KL={kl}, const={const}")
        print(f"Bound={bound}")

    return bound 

n_train, n_test = 10, 5
delta = 0.01
M = 4
rnd_seed = 17032021

np.random.seed(rnd_seed)
random.seed(rnd_seed)
jkey = jrand.PRNGKey(rnd_seed)

train_x, train_y, test_x, test_y = load_normals(n_train, n_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

d = train_x.shape[1]

# predictors, num_preds = uniform_decision_stumps(M, d, train_x.min(0), train_x.max(0))
predictors, num_preds = custom_decision_stumps(np.zeros((2, 2)), np.array([[1, -1], [1, -1]]))

beta = jnp.ones(num_preds) * 0.1 # prior
alpha = jrand.uniform(jkey, shape=(num_preds,), minval=0.01, maxval=2) # posterior

test_error = risk(alpha, predictors, (test_x, test_y))

# print(f"Initial McAllester bound, for delta={delta}, n={n_train}")
# mcallester_bound(alpha, beta, delta, predictors, (train_x, train_y), verbose=True)

print("Initial test error:", test_error)

# alpha_mca = batch_gradient_descent(mcallester_bound, alpha, (beta, delta, predictors, (train_x, train_y)), lr=0.1, num_iters=1000)

# print(f"Optimized McAllester bound, for delta={delta}, n={n_train}")
# mcallester_bound(alpha_mca, beta, delta, predictors, (train_x, train_y), verbose=True)

# print("Optimized test error:", test_error)

print("Optimize only empirical risk")
alpha_err = batch_gradient_descent(risk, alpha, (predictors, (train_x, train_y)), lr=1, num_iters=200)
# import pdb; pdb.set_trace()
test_error = risk(alpha_err, predictors, (test_x, test_y))

print("Optimized test error:", test_error)