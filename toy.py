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

@hydra.main(config_name='config/toy.yaml')
def main(cfg):

    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    jkey = jrand.PRNGKey(cfg.training.seed)

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.training.risk}/{cfg.model.pred}/{cfg.dataset.distr}/prior={cfg.model.prior}/lr={cfg.training.lr}/seed={cfg.training.seed}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve())

    n_values = [1, 10, 100, 500, 1000]
    results = {"time": [], f"{cfg.bound.type}-bound": [], "test-error": [], "n-values": n_values}
    for n in n_values:

        if cfg.dataset.distr == "normals":
            train_x, train_y, test_x, test_y = load(cfg.dataset.distr, n, cfg.dataset.N_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

        else:
            train_x, train_y, test_x, test_y = load(cfg.dataset.distr, n, cfg.dataset.N_test)

        if cfg.model.pred == "stumps-uniform":
            predictors, cfg.model.M = uniform_decision_stumps(cfg.model.M, 2, train_x.min(0), train_x.max(0))

        elif cfg.model.pred == "stumps-optimal":
            predictors, cfg.model.M = custom_decision_stumps(np.zeros((2, 2)), np.array([[1, -1], [1, -1]]))

        # use exp(log(alpha)) for numerical stability
        beta = jnp.log(jnp.ones(cfg.model.M) * cfg.model.prior) # prior
        alpha = jnp.log(jrand.uniform(jkey, shape=(cfg.model.M,), minval=0.01, maxval=2)) # posterior

        # init train-eval monitoring 
        monitor = MonitorMV(SAVE_DIR)

        if cfg.training.risk == "exact":

            print("Optimize empirical risk")
            cost, params = risk, (predictors, (train_x, train_y))

        elif cfg.training.risk == "MC":

            print("Optimize empirical sigmoid risk")
            cost, params = approximated_risk, (predictors, (train_x, train_y), sigmoid_loss, jkey)

        t1 = time()
        alpha_opt = batch_gradient_descent(cost, alpha, params, lr=cfg.training.lr, num_iters=int(cfg.training.iter), monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = risk(alpha_opt, predictors, (test_x, test_y))

        print("Optimized test error:", test_error)

        b = mcallester_bound(alpha_opt, beta, cfg.bound.delta, predictors, (train_x, train_y), verbose=True)

        results["time"].append(t2-t1)
        results[f"{cfg.bound.type}-bound"].append(b)
        results["test-error"].append(test_error)

        monitor.write(cfg.training.iter, end={"test-error": test_error, "train-time": t2-t1, f"{cfg.bound.type}-bound": b})

        monitor.close()

    np.save(SAVE_DIR / "results.npy", results)

if __name__ == "__main__":
    main()