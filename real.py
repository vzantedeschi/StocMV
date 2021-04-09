import hydra
from time import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.bounds import BOUNDS
from core.losses import sigmoid_loss
from core.monitors import MonitorMV
from core.optimization import train_stochastic, evaluate
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.stochastic_mv import MajorityVote, uniform_decision_stumps, custom_decision_stumps

@hydra.main(config_path='config/real.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.name}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/M={cfg.model.M}/prior={cfg.model.prior}/lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/seeds={cfg.training.seed}-{cfg.training.seed+cfg.num_trials}/"

    SAVE_DIR = Path(SAVE_DIR)

    if cfg.training.risk == "MC":
        SAVE_DIR = SAVE_DIR / f"MC={cfg.training.MC_draws}"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve()) 

    data = Dataset(cfg.dataset.name, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data")     

    train_errors, test_errors, bounds, times = [], [], [], []
    for i in range(cfg.num_trials):
        
        deterministic(cfg.training.seed+i)

        if cfg.model.pred == "stumps-uniform":
            predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

        else:
            raise NotImplementedError("Only stumps-uniform supported atm")

        beta = torch.ones(M) * cfg.model.prior # prior

        model = MajorityVote(predictors, beta, mc_draws=cfg.training.MC_draws)

        bound = None
        if cfg.training.opt_bound:

            print(f"Optimize {cfg.bound.type} bound")

            if cfg.bound.stochastic:
                print("Evaluate bound regularizations over mini-batch")
                bound = lambda n, m, r: BOUNDS[cfg.bound.type](n, m, r, cfg.bound.delta)

            else:
                print("Evaluate bound regularizations over whole train+val set")
                bound = lambda n, m, r: BOUNDS[cfg.bound.type](len(data.X_train) + len(data.X_valid), m, r, cfg.bound.delta)

        loss = None
        if cfg.training.risk == "MC":

            print("with approximated risk, using sigmoid loss")
            loss = sigmoid_loss

        trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)
        valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers, shuffle=False)
        testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers, shuffle=False)

        trainvalloader = DataLoader(TorchDataset(np.vstack([data.X_train, data.X_valid]), np.vstack([data.y_train, data.y_valid])), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers)

        monitor = MonitorMV(SAVE_DIR)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)
        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

        best_val_error = float("inf")
        best_e = -1
        no_improv = 0

        t1 = time()
        for e in range(cfg.training.num_epochs):
            train_stochastic(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor)

            val_error = evaluate(valloader, model, epoch=e, monitor=monitor)
            train_error = evaluate(trainvalloader, model, epoch=e, bounds={cfg.bound.type: bound}, monitor=monitor, tag="train-val")
            print(f"Epoch {e}: {val_error['error']}\n")
            
            no_improv += 1
            if val_error['error'] < best_val_error:
                best_val_error = val_error['error']
                best_train_stats = train_error
                best_e = e
                best_model = deepcopy(model)
                no_improv = 0

            # reduce learning rate if needed
            lr_scheduler.step(val_error['error'])

            if no_improv == cfg.training.num_epochs // 4:
                break

        t2 = time()

        test_error = evaluate(testloader, best_model, epoch=e, tag="test")

        print(f"Test error: {test_error['error']}; {cfg.bound.type} bound: {best_train_stats[cfg.bound.type]}\n")
        
        train_errors.append(best_train_stats['error'])
        test_errors.append(test_error['error'])
        bounds.append(best_train_stats[cfg.bound.type])
        
        monitor.close()
        
    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds))})

if __name__ == "__main__":
    main()
