import hydra
from time import time
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam

from core.bounds import BOUNDS
from core.losses import sigmoid_loss
from core.monitors import MonitorMV
from core.optimization import train_batch
from core.utils import deterministic
from data.toy_datasets import load
from models.stochastic_mv import MajorityVote, uniform_decision_stumps, custom_decision_stumps

@hydra.main(config_path='config/toy.yaml')
def main(cfg):

    NUM_TRIALS = 10
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

        train_x, train_y, test_x, test_y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float(), torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()

        # use exp(log(alpha)) for numerical stability
        beta = torch.ones(M) * cfg.model.prior # prior

        model = MajorityVote(predictors, beta)

        bound = None
        if cfg.training.opt_bound:

            print(f"Optimize {cfg.bound.type} bound")
            bound = lambda d, m, r: BOUNDS[cfg.bound.type](d, m, r, cfg.bound.delta)

        loss = None
        if cfg.training.risk == "MC":

            print("with approximated risk, using sigmoid loss")
            loss = sigmoid_loss

        # get voter predictions
        train_data = train_y.unsqueeze(1), predictors(train_x)
        test_data = test_y.unsqueeze(1), predictors(test_x)

        monitor = MonitorMV(SAVE_DIR)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)

        t1 = time()
        train_batch(train_data, model, optimizer, bound=bound, loss=loss, nb_iter=cfg.training.iter, monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = model.risk(test_data)
        train_error = model.risk(train_data)

        print(f"Test error: {test_error.item()}")

        b = float(BOUNDS[cfg.bound.type](len(train_data[0]), model, train_error, cfg.bound.delta, verbose=True))
        
        train_errors.append(train_error.item())
        test_errors.append(test_error.item())
        bounds.append(b)
    
    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds))})

if __name__ == "__main__":
    main()
