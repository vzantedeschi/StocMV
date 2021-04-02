import hydra
from time import time
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam

from core.metrics import Metrics
from core.monitors import MonitorMV
from core.optimization import train_batch
from core.utils import deterministic
from data.toy_datasets import load
from learner.c_bound_joint_learner import CBoundJointLearner
from models.stochastic_mv import MajorityVote, uniform_decision_stumps, custom_decision_stumps

@hydra.main(config_path='config/toy_oracle.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.distr}/cbound/joint/optimize-bound=True/{cfg.model.pred}/M={cfg.model.M}/prior=uniform/lr={cfg.training.lr}/seeds={cfg.training.seed}-{cfg.training.seed+cfg.num_trials}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve()) 
    
    train_errors, test_errors, bounds = [], [], []
    for i in range(cfg.num_trials):
        
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

        prior = torch.ones(M) / M # uniform prior

        # get voter predictions
        train_data = train_y.unsqueeze(1), predictors(train_x)
        test_data = test_y.unsqueeze(1), predictors(test_x)

        model = MajorityVote(predictors, prior, distr="categorical")
        learner = CBoundJointLearner(delta=cfg.bound.delta)

        monitor = MonitorMV(SAVE_DIR, normalize=True)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)

        t1 = time()
        train_batch(train_data, model, optimizer, learner=learner, nb_iter=cfg.training.iter, monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = model.risk(test_data)
        train_error = model.risk(train_data)

        print(f"Test error: {test_error.item()}")
        
        if cfg.bound.type == "seeger":
            b = Metrics("CBoundSeeger", model, delta=cfg.bound.delta).fit(*train_data)
        elif cfg.bound.type == "mcallester":
            b = Metrics("CBoundMcAllester", model, delta=cfg.bound.delta).fit(*train_data)
        else:
            raise NotImplementedError
        
        train_errors.append(train_error.item())
        test_errors.append(test_error.item())
        bounds.append(b)
    
        monitor.close()

    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds))})


if __name__ == "__main__":
    main()
