import jax.numpy as jnp
import jax.random as jrand

from datasets import *
from dirichlet import *
from optimization import batch_gradient_descent
from predictors import uniform_decision_stumps

def mcallester_bound(alpha, beta, delta, predictors, data, verbose=False):

    n = len(data[0])
    
    kl = KL(alpha, beta)
    tr = risk(alpha, predictors, data)
    const = jnp.log(2 * (n**0.5) / delta)

    bound = tr + ((kl + const) / 2 / n)**0.5

    if verbose:
        print(f"True risk={tr}, KL={kl}, const={const}")
        print(f"Bound={bound}")

    return bound 

n_train, n_test = 1000, 500
delta = 0.01
M = 16
rnd_seed = 17032021
jkey = jrand.PRNGKey(rnd_seed)

beta = jnp.ones(M) * 0.1 # prior
# alpha = jnp.ones(M) * 0.1 # posterior

alpha = jrand.uniform(jkey, shape=(M,), minval=0.01, maxval=2) # posterior

train_x, train_y, test_x, test_y = load_normals(n_train, n_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

predictors = uniform_decision_stumps(M, train_x.shape[1], train_x.min(), train_x.max())

test_error = risk(alpha, predictors, (test_x, test_y))

# print(f"Initial McAllester bound, for delta={delta}, n={n_train}")
# mcallester_bound(alpha, beta, delta, predictors, (train_x, train_y), verbose=True)

print("Initial test error:", test_error)

# alpha_mca = batch_gradient_descent(mcallester_bound, alpha, (beta, delta, predictors, (train_x, train_y)), lr=0.1, num_iters=1000)

# print(f"Optimized McAllester bound, for delta={delta}, n={n_train}")
# mcallester_bound(alpha_mca, beta, delta, predictors, (train_x, train_y), verbose=True)

# print("Optimized test error:", test_error)

print("Optimize only empirical risk")
alpha_err = batch_gradient_descent(risk, alpha, (predictors, (train_x, train_y)), lr=0.1, num_iters=100)
test_error = risk(alpha_err, predictors, (test_x, test_y))

print("Optimized test error:", test_error)