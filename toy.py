import hydra

from time import time

from pathlib import Path

import numpy as np
import random

import jax.numpy as jnp
import jax.random as jrand

from bounds import *
from datasets import load
from dirichlet import *
from loss import sigmoid_loss
from monitors import MonitorMV
from optimization import batch_gradient_descent
from predictors import uniform_decision_stumps, custom_decision_stumps

@hydra.main(config_path='config/toy.yaml')
def main(cfg):

    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    jkey = jrand.PRNGKey(cfg.training.seed)

    if cfg.dataset.distr == "normals":
        
        train_x, train_y, test_x, test_y = load(cfg.dataset.distr, cfg.dataset.N_train, cfg.dataset.N_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

    else:
        train_x, train_y, test_x, test_y = load(cfg.dataset.distr, cfg.dataset.N_train, cfg.dataset.N_test)

    d = train_x.shape[1]

    if cfg.model.pred == "stumps-uniform":
        predictors, cfg.model.M = uniform_decision_stumps(cfg.model.M, d, train_x.min(0), train_x.max(0))

    elif cfg.model.pred == "stumps-optimal":
        predictors, cfg.model.M = custom_decision_stumps(np.zeros((2, 2)), np.array([[1, -1], [1, -1]]))

    # use exp(log(alpha)) for numerical stability
    beta = jnp.log(jnp.ones(cfg.model.M) * cfg.model.prior) # prior
    alpha = jnp.log(jrand.uniform(jkey, shape=(cfg.model.M,), minval=0.01, maxval=2)) # posterior

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.training.risk}/{cfg.model.pred}/{cfg.dataset.distr}/N={cfg.dataset.N_train}/M={cfg.model.M}/prior={cfg.model.prior}/lr={cfg.training.lr}/seed={cfg.training.seed}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve())

    test_error = risk(alpha, predictors, (test_x, test_y))
    print("Initial test error:", test_error)

    print(f"Initial McAllester bound, for delta={cfg.bound.delta}, n={cfg.dataset.N_train}")
    mcallester_bound(alpha, beta, cfg.bound.delta, predictors, (train_x, train_y), verbose=True)

    # t1 = time()
    # alpha_mca = batch_gradient_descent(mcallester_bound, alpha, (beta, cfg.bound.delta, predictors, (train_x, train_y)), lr=1, num_iters=10)
    # t2 = time()

    # print(f"Optimized McAllester bound, for cfg.bound.delta={cfg.bound.delta}, n={n_train}")
    # mcallester_bound(alpha_mca, beta, cfg.bound.delta, predictors, (train_x, train_y), verbose=True)
    # print(f"{t2-t1}s for 10 iterations")
    # test_error = risk(alpha_err, predictors, (test_x, test_y))

    # print("Optimized test error:", test_error)

    # init train-eval monitoring 
    monitor = MonitorMV(SAVE_DIR)

    if cfg.training.risk == "exact":

        print("Optimize only empirical risk")
        t1 = time()
        alpha_err = batch_gradient_descent(risk, alpha, (predictors, (train_x, train_y)), lr=cfg.training.lr, num_iters=cfg.training.iter, monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = risk(alpha_err, predictors, (test_x, test_y))

        print("Optimized test error:", test_error)

        b = mcallester_bound(alpha_err, beta, cfg.bound.delta, predictors, (train_x, train_y), verbose=True)

    elif cfg.training.risk == "MC":
        print("Optimize only sigmoid empirical risk")
        t1 = time()
        alpha_appr = batch_gradient_descent(approximated_risk, alpha, (predictors, (train_x, train_y), sigmoid_loss, jkey), lr=cfg.training.lr, num_iters=cfg.training.iter, monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = risk(alpha_appr, predictors, (test_x, test_y))

        print("Optimized test error:", test_error)

        b = mcallester_bound(alpha_appr, beta, cfg.bound.delta, predictors, (train_x, train_y), verbose=True)

    monitor.write(cfg.training.iter, end={"test-error": test_error, "train-time": t2-t1, f"{cfg.bound.type}-bound": b})

    monitor.close()

if __name__ == "__main__":
    main()