import hydra

from time import time

from pathlib import Path

import numpy as np
import random

import jax.numpy as jnp
import jax.random as jrand

from bounds import *
from datasets import load
import categorical as cat
from loss import moment_loss
from monitors import MonitorMV
from optimization import batch_gradient_descent
from predictors import uniform_decision_stumps, custom_decision_stumps

@hydra.main(config_path='config/toy_oracle.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.distr}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/{cfg.model.pred}/M={cfg.model.M}/prior=uniform/lr={cfg.training.lr}/seed={cfg.training.seed}-{cfg.training.seed+10}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("results will be saved in:", SAVE_DIR.resolve())
    
    monitor = None
    test_errors, train_errors, bounds = [], [], []
    for i in range(10):
        
        np.random.seed(cfg.training.seed+i)
        random.seed(cfg.training.seed+i)
        jkey = jrand.PRNGKey(cfg.training.seed+i)
    
        if cfg.dataset.distr == "normals":
            train_x, train_y, test_x, test_y = load(cfg.dataset.distr, cfg.dataset.N_train, cfg.dataset.N_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

        else:
            train_x, train_y, test_x, test_y = load(cfg.dataset.distr, cfg.dataset.N_train, cfg.dataset.N_test)

        if cfg.model.pred == "stumps-uniform":
            predictors, M = uniform_decision_stumps(cfg.model.M, 2, train_x.min(0), train_x.max(0))

        elif cfg.model.pred == "stumps-optimal":
            predictors, M = custom_decision_stumps(np.zeros((2, 2)), np.array([[1, -1], [1, -1]]))

        beta = jnp.log(jnp.ones(M) / M) # uniform prior
        
        loss = lambda x, y, z: moment_loss(x, y, z, order=cfg.training.risk)
        if cfg.training.opt_bound:

            print(f"Optimize {cfg.bound.type} bound")

            print(f"Using {cfg.training.risk} order loss")
            cost, params = BOUNDS[cfg.bound.type], (cat.risk_upper_bound, beta, cfg.bound.delta, (loss,), 2**cfg.training.risk)
        
        else:

            print(f"Optimize train risk")

            print(f"Using {cfg.training.risk} order loss")
            cost, params = cat.risk_upper_bound, (loss,)
    

        alpha = jrand.uniform(jkey, shape=(M,), minval=0, maxval=10) # posterior
        alpha /= alpha.sum() # has to sum to 1
        alpha = jnp.log(alpha)

        # get voter predictions
        train_data = train_x, train_y[..., None], predictors(train_x)
        test_data = test_x, test_y[..., None], predictors(test_x)

        # init train-eval monitoring 
        # monitor = MonitorMV(SAVE_DIR)
        t1 = time()
        alpha_opt = batch_gradient_descent(train_data, alpha, cost, params, lr=cfg.training.lr, num_iters=int(cfg.training.iter), monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = float(cat.risk(test_data, alpha_opt))
        train_error = float(cat.risk(train_data, alpha_opt))

        print(f"Test error: {test_error}")

        b = float(BOUNDS[cfg.bound.type](train_data, alpha_opt, cat.risk_upper_bound, beta, cfg.bound.delta, (loss,), 2**cfg.training.risk, verbose=True))
        
        test_errors.append(test_error)
        train_errors.append(train_error)
        bounds.append(b)

    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds))})

if __name__ == "__main__":
    main()
