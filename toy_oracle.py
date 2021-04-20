import hydra
from time import time
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam

from core.bounds import BOUNDS
from core.losses import moment_loss, exp_loss
from core.monitors import MonitorMV
from core.optimization import train_batch
from core.utils import deterministic
from data.datasets import Dataset
from models.majority_vote import MajorityVote
from models.random_forest import decision_trees
from models.stumps import uniform_decision_stumps

from graphics.plot_predictions import plot_2D
import matplotlib.pyplot as plt

@hydra.main(config_path='config/toy_oracle.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.distr}/{cfg.dataset.N_train}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/{cfg.model.pred}/M={cfg.model.M}/prior=uniform/lr={cfg.training.lr}/seeds={cfg.training.seed}-{cfg.training.seed+cfg.num_trials}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve()) 
    
    train_errors, test_errors, bounds, risks = [], [], [], []
    for i in range(cfg.num_trials):
        
        deterministic(cfg.training.seed+i)

        data = Dataset(cfg.dataset.distr, n_train=cfg.dataset.N_train, n_test=cfg.dataset.N_test)     

        if cfg.model.pred == "stumps-uniform":
            predictors, M = uniform_decision_stumps(cfg.model.M, 2, data.X_train.min(0), data.X_train.max(0))

        elif cfg.model.pred == "stumps-optimal":
            predictors, M = custom_decision_stumps(torch.zeros((2, 2)), torch.tensor([[1, -1], [1, -1]]))

        elif cfg.model.pred == "rf": # random forest

            if cfg.model.tree_depth == "None":
                cfg.model.tree_depth = None

            predictors, M = decision_trees(cfg.model.M, (data.X_train, data.y_train[:, 0]), max_samples=cfg.model.boostrap, max_depth=cfg.model.tree_depth)

        else:
            raise NotImplementedError("model.pred should be one the following: [stumps-uniform, stumps-uniform, rf]")

        train_x, train_y, test_x, test_y = torch.from_numpy(data.X_train).float(), torch.from_numpy(data.y_train).float(), torch.from_numpy(data.X_test).float(), torch.from_numpy(data.y_test).float()

        prior = torch.ones(M) / M # uniform prior

        model = MajorityVote(predictors, prior, distr="categorical")
        
        if cfg.training.risk in [1, 2]:
            loss = lambda x, y, z: moment_loss(x, y, z, order=cfg.training.risk)
            coeff = 2**cfg.training.risk

        elif cfg.training.risk == "exp":
            loss = lambda x, y, z: exp_loss(x, y, z, c=cfg.training.risk_c)
            coeff = np.exp(cfg.training.risk_c / 2) - 1

        else:
            raise NotImplementedError()

        bound = None
        if cfg.training.opt_bound:

            print(f"Optimize {cfg.bound.type} bound")
            bound = lambda d, m, r: BOUNDS[cfg.bound.type](d, m, r, cfg.bound.delta, coeff=coeff)

        # get voter predictions
        train_data = train_y, predictors(train_x)
        test_data = test_y, predictors(test_x)

        monitor = MonitorMV(SAVE_DIR, normalize=True)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)

        t1 = time()
        train_batch(train_data, model, optimizer, bound=bound, loss=loss, nb_iter=cfg.training.iter, monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = model.risk(test_data)
        train_error = model.risk(train_data)
        train_risk = model.risk(train_data, loss)

        print(f"Test error: {test_error.item()}")
        
        b = float(BOUNDS[cfg.bound.type](len(train_data[0]), model, train_risk, cfg.bound.delta, coeff=coeff, verbose=True))
        
        train_errors.append(train_error.item())
        test_errors.append(test_error.item())
        bounds.append(b)
        risks.append(train_risk.item())
        
        monitor.close()
        
        plot_2D(data, model, bound=b)

        plt.title(f"{cfg.model.pred}, {cfg.bound.type} bound, M={M}")

        plt.savefig(SAVE_DIR / f"{cfg.dataset.distr}.pdf", bbox_inches='tight', transparent=True)
        plt.clf()

    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "train-risk": (np.mean(risks), np.std(risks))})


if __name__ == "__main__":
    main()
