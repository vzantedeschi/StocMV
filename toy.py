import numpy as np
import random

import jax.numpy as jnp
import jax.random as jrand

from bounds import *
from datasets import *
from dirichlet import *
from loss import sigmoid_loss
from optimization import batch_gradient_descent
from predictors import uniform_decision_stumps, custom_decision_stumps

from time import time

iters = 100
n_train, n_test = 100, 100
delta = 0.01
M = 6
rnd_seed = 17032021

np.random.seed(rnd_seed)
random.seed(rnd_seed)
jkey = jrand.PRNGKey(rnd_seed)

train_x, train_y, test_x, test_y = load_normals(n_train, n_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

d = train_x.shape[1]

# predictors, num_preds = uniform_decision_stumps(M, d, train_x.min(0), train_x.max(0))
predictors, num_preds = custom_decision_stumps(np.zeros((2, 2)), np.array([[1, -1], [1, -1]]))

# use exp(log(alpha)) for numerical stability
beta = jnp.log(jnp.ones(num_preds) * 0.1) # prior
alpha = jnp.log(jrand.uniform(jkey, shape=(num_preds,), minval=0.01, maxval=2)) # posterior

test_error = risk(alpha, predictors, (test_x, test_y))
print("Initial test error:", test_error)

print(f"Initial McAllester bound, for delta={delta}, n={n_train}")
mcallester_bound(alpha, beta, delta, predictors, (train_x, train_y), verbose=True)

# t1 = time()
# alpha_mca = batch_gradient_descent(mcallester_bound, alpha, (beta, delta, predictors, (train_x, train_y)), lr=1, num_iters=10)
# t2 = time()

# print(f"Optimized McAllester bound, for delta={delta}, n={n_train}")
# mcallester_bound(alpha_mca, beta, delta, predictors, (train_x, train_y), verbose=True)
# print(f"{t2-t1}s for 10 iterations")
# test_error = risk(alpha_err, predictors, (test_x, test_y))

# print("Optimized test error:", test_error)

print("Optimize only sigmoid empirical risk")
t1 = time()
alpha_appr = batch_gradient_descent(approximated_risk, alpha, (predictors, (train_x, train_y), sigmoid_loss, jkey), lr=1, num_iters=iters)
t2 = time()
print(f"{t2-t1}s for {iters} iterations")

test_error = risk(alpha_appr, predictors, (test_x, test_y))

print("Optimized test error:", test_error)

mcallester_bound(alpha_appr, beta, delta, predictors, (train_x, train_y), verbose=True)

print("Optimize only empirical risk")
t1 = time()
alpha_err = batch_gradient_descent(risk, alpha, (predictors, (train_x, train_y)), lr=1, num_iters=iters)
t2 = time()
print(f"{t2-t1}s for {iters} iterations")

test_error = risk(alpha_err, predictors, (test_x, test_y))

print("Optimized test error:", test_error)

mcallester_bound(alpha_err, beta, delta, predictors, (train_x, train_y), verbose=True)