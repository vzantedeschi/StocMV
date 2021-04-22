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
from data.datasets import Dataset
from models.majority_vote import MultipleMajorityVote, MajorityVote
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps, custom_decision_stumps

from graphics.plot_predictions import plot_2D
import matplotlib.pyplot as plt

@hydra.main(config_path='config/toy.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.distr}/{cfg.dataset.N_train}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/{cfg.model.pred}/M={cfg.model.M}/prior={cfg.model.prior}/lr={cfg.training.lr}/seeds={cfg.training.seed}-{int(cfg.training.seed)+cfg.num_trials}/"

    SAVE_DIR = Path(SAVE_DIR)

    if cfg.training.risk == "MC":
        SAVE_DIR = SAVE_DIR / f"MC={cfg.training.MC_draws}"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve()) 
    
    train_errors, test_errors, bounds, times = [], [], [], []
    for i in range(cfg.num_trials):
        
        deterministic(int(cfg.training.seed)+i)

        data = Dataset(cfg.dataset.distr, n_train=cfg.dataset.N_train, n_test=cfg.dataset.N_test) 

        m = None
        if cfg.model.pred == "stumps-uniform":
            predictors, M = uniform_decision_stumps(cfg.model.M, 2, data.X_train.min(0), data.X_train.max(0))

        elif cfg.model.pred == "stumps-custom":
            predictors, M = custom_decision_stumps(torch.zeros((2, 2)), torch.tensor([[1, -1], [1, -1]]))

        elif cfg.model.pred == "rf": # random forest

            if cfg.model.tree_depth == "None":
                cfg.model.tree_depth = None

            m = int(cfg.dataset.N_train*cfg.model.m) # number of points for learning first prior
            predictors, M = two_forests(cfg.model.M, m, data.X_train, data.y_train[:, 0], max_samples=cfg.model.bootstrap, max_depth=cfg.model.tree_depth, binary=True)

        else:
            raise NotImplementedError("model.pred should be one the following: [stumps-uniform, stumps-custom, rf]")

        train_x, train_y, test_x, test_y = torch.from_numpy(data.X_train).float(), torch.from_numpy(data.y_train).float(), torch.from_numpy(data.X_test).float(), torch.from_numpy(data.y_test).float()

        monitor = MonitorMV(SAVE_DIR)

        betas = [torch.ones(M) * cfg.model.prior for p in predictors] # prior

        if len(predictors) == 2:
            model = MultipleMajorityVote(predictors, betas, (cfg.model.m, 1-cfg.model.m), mc_draws=cfg.training.MC_draws)
        
        else:
            model = MajorityVote(predictors, betas, mc_draws=cfg.training.MC_draws)

        bound = None
        if cfg.training.opt_bound:

            print(f"Optimize {cfg.bound.type} bound")
            bound = lambda n, model, risk: BOUNDS[cfg.bound.type](n, model, risk, cfg.bound.delta, m=m, monitor=monitor)

        loss = None
        if cfg.training.risk == "MC":

            print("with approximated risk, using sigmoid loss")
            loss = sigmoid_loss

        # get voter predictions
        if m: 
            # use first m data for learning the second posterior, and the remainder for the first one
            train_data = [(train_y[m:], p(train_x[m:])) for p in predictors]
            # test both posterior on entire test set
            test_data = [(test_y, p(test_x)) for p in predictors]

        else:
            train_data = train_y, predictors(train_x)
            test_data = test_y, predictors(test_x)

        optimizer = Adam(model.parameters(), lr=cfg.training.lr)

        t1 = time()
        train_batch(cfg.dataset.N_train, train_data, model, optimizer, bound=bound, loss=loss, nb_iter=cfg.training.iter, monitor=monitor)
        t2 = time()
        print(f"{t2-t1}s for {cfg.training.iter} iterations")

        test_error = model.risk(test_data)
        train_error = model.risk(train_data)

        print(f"Test error: {test_error.item()}")

        b = float(BOUNDS[cfg.bound.type](cfg.dataset.N_train, model, train_error, cfg.bound.delta, m=m, verbose=True))
        
        train_errors.append(train_error.item())
        test_errors.append(test_error.item())
        bounds.append(b)
        times.append(t2-t1)
        
        monitor.close()

        plot_2D(data, model, bound=b)

        plt.title(f"{cfg.model.pred}, {cfg.bound.type} bound, M={M}")

        plt.savefig(SAVE_DIR / f"{cfg.dataset.distr}.pdf", bbox_inches='tight', transparent=True)
        plt.clf()
        
    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times))})

if __name__ == "__main__":
    main()
