import hydra

from time import time

from pathlib import Path

import numpy as np
import random

from learner.bounds import BOUNDS
from data.datasets import load
from core.losses import sigmoid_loss
from core.monitors import MonitorMV
from core.utils import deterministic
from models.stochastic_mv import MajorityVote, uniform_decision_stumps, custom_decision_stumps

@hydra.main(config_path='config/toy.yaml')
def main(cfg):

    NUM_TRIALS = 1
    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.distr}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/{cfg.model.pred}/M={cfg.model.M}/prior={cfg.model.prior}/lr={cfg.training.lr}/seeds={cfg.training.seed}-{cfg.training.seed+NUM_TRIALS}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve()) 
    
    train_errors, test_errors, bounds = [], [], []
    for i in range(NUM_TRIALS):
        
        deterministic(cfg.training.seed+i)

        if cfg.dataset.distr == "normals":
            train_x, train_y, test_x, test_y = load(cfg.dataset.distr, cfg.dataset.N_train, cfg.dataset.N_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

        else:
            train_x, train_y, test_x, test_y = load(cfg.dataset.distr, cfg.dataset.N_train, cfg.dataset.N_test)

        if cfg.model.pred == "stumps-uniform":
            predictors, M = uniform_decision_stumps(cfg.model.M, 2, train_x.min(0), train_x.max(0))

        elif cfg.model.pred == "stumps-optimal":
            predictors, M = custom_decision_stumps(np.zeros((2, 2)), np.array([[1, -1], [1, -1]]))

        # use exp(log(alpha)) for numerical stability
        beta = torch.ones(M) * cfg.model.prior # prior

        model = MajorityVote(predictors, beta)

        if cfg.training.opt_bound:

            print(f"Optimize {cfg.bound.type} bound")

        if cfg.training.risk == "exact":

            print("Using 01-loss")
            cost, params = BOUNDS[cfg.bound.type], (model, cfg.bound.delta, ())

        elif cfg.training.risk == "MC":

            print("Using sigmoid loss")
            cost, params = BOUNDS[cfg.bound.type], (model, cfg.bound.delta, (sigmoid_loss, ))
        
        else:
            raise NotImplementedError

        # get voter predictions
        train_data = train_x, train_y[..., None], predictors(train_x)
        test_data = test_x, test_y[..., None], predictors(test_x)

        # init train-eval monitoring 
        # monitor = None
        monitor = MonitorMV(SAVE_DIR)

        t1 = time()
        alpha_opt = batch_gradient_descent(train_data, alpha, cost, params, lr=cfg.training.lr, num_iters=int(cfg.training.iter), monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = float(diri.risk(test_data, alpha_opt))
        train_error = float(diri.risk(train_data, alpha_opt))

        print(f"Test error: {test_error}")

        b = float(BOUNDS[cfg.bound.type](train_data, alpha_opt, diri.risk, beta, cfg.bound.delta, (), verbose=True))
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        bounds.append(b)
    
    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds))})

if __name__ == "__main__":
    main()
