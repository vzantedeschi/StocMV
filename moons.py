from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import jax.numpy as jnp

from dirichlet import *
from predictors import decision_stumps

def mcallester_bound(alpha, beta, delta, data, predictors, verbose=True):

    n = len(data[0])
    
    kl = KL(alpha, beta)
    tr = risk(alpha, predictors, data)
    const = jnp.log(2 * (n**0.5) / delta)

    bound = tr + ((kl + const) / 2 / n)**0.5

    if verbose:
        print(f"True risk={tr}, KL={kl}, const={const}")
        print(f"Bound={bound}")

    return bound 

n = 2000
delta = 0.1

train_x, train_y = datasets.make_moons(n_samples=n, noise=.05)
test_x, test_y = datasets.make_moons(n_samples=500, noise=.05)

train_y[train_y == 0] = -1
test_y[test_y == 0] = -1

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

predictors = decision_stumps(8, train_x.shape[1], train_x.min(), train_x.max())

beta = jnp.ones(len(predictors))
alpha = jnp.ones(len(predictors)) * 2

test_error = risk(alpha, predictors, (test_x, test_y))

print(f"McAllester bound, for delta={delta}, n={n}")
b = mcallester_bound(alpha, beta, 0.1, (train_x, train_y), predictors)

print("Test error:", test_error)
