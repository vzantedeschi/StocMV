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
from core.losses import moment_loss
from core.monitors import MonitorMV
from core.optimization import train_stochastic, evaluate
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.stochastic_mv import MajorityVote, uniform_decision_stumps, custom_decision_stumps

@hydra.main(config_path='config/real_oracle.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.name}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/M={cfg.model.M}/prior=uniform/lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/seeds={cfg.training.seed}-{cfg.training.seed+cfg.num_trials}/"

    SAVE_DIR = Path(SAVE_DIR)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve()) 

    data = Dataset(cfg.dataset.name, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data")     

    train_errors, test_errors, bounds, times, risks = [], [], [], [], []
    for i in range(cfg.num_trials):
        
        deterministic(cfg.training.seed+i)

        if cfg.model.pred == "stumps-uniform":
            predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

        else:
            raise NotImplementedError("Only stumps-uniform supported atm")

        beta = torch.ones(M) / M # prior

        model = MajorityVote(predictors, beta, mc_draws=cfg.training.MC_draws, distr="categorical")

        loss = lambda x, y, z: moment_loss(x, y, z, order=cfg.training.risk)

        bound = None
        if cfg.training.opt_bound:

            print(f"Optimize {cfg.bound.type} bound")
            bound = lambda d, m, r: BOUNDS[cfg.bound.type](d, m, r, cfg.bound.delta, coeff=2**cfg.training.risk)

        trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)
        valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers, shuffle=False)
        testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers, shuffle=False)

        monitor = MonitorMV(SAVE_DIR, normalize=True)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)
        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

        best_val_loss = float("inf")
        best_e = -1
        no_improv = 0

        t1 = time()
        for e in range(cfg.training.num_epochs):
            train_stochastic(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor)

            val_loss = evaluate(valloader, model, epoch=e, monitor=monitor)
            print(f"Epoch {e}: {val_loss['error']}\n")
            
            no_improv += 1
            if val_loss['error'] < best_val_loss:
                best_val_loss = val_loss['error']
                best_e = e
                best_model = deepcopy(model)
                no_improv = 0

            # reduce learning rate if needed
            lr_scheduler.step(val_loss['error'])

            if no_improv == cfg.training.num_epochs // 4:
                break

        t2 = time()

        trainvalloader = DataLoader(TorchDataset(np.vstack([data.X_train, data.X_valid]), np.vstack([data.y_train, data.y_valid])), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers)

        test_error = evaluate(testloader, best_model, epoch=e, tag="test")
        train_error = evaluate(trainvalloader, best_model, epoch=e, tag="train-val")
        train_risk = evaluate(trainvalloader, best_model, epoch=e, bounds={cfg.bound.type: bound}, loss=loss, tag="train-val")

        print(f"Test error: {test_error['error']}; {cfg.bound.type} bound: {train_risk[cfg.bound.type]}\n")
        
        train_errors.append(train_error['error'])
        test_errors.append(test_error['error'])
        bounds.append(train_risk[cfg.bound.type])
        risks.append(train_risk['error'])
        
        monitor.close()
        
    np.save(SAVE_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "train-risk": (np.mean(risks), np.std(risks))})

if __name__ == "__main__":
    main()
