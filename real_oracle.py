import hydra
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.bounds import BOUNDS
from core.losses import moment_loss
from core.monitors import MonitorMV
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MajorityVote
from models.stumps import uniform_decision_stumps

from training_routines import stochastic_routine

@hydra.main(config_path='config/real_oracle.yaml')
def main(cfg):

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/M={cfg.model.M}/prior=uniform/lr={cfg.training.lr}/batch-size={cfg.training.batch_size}"

    ROOT_DIR = Path(ROOT_DIR)

    print("results will be saved in:", ROOT_DIR.resolve()) 

    data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data")     
    if cfg.model.pred == "stumps-uniform":
        predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

    else:
        raise NotImplementedError("Only stumps-uniform supported atm")

    beta = torch.ones(M) / M # prior

    loss = lambda x, y, z: moment_loss(x, y, z, order=cfg.training.risk)

    bound = None
    if cfg.training.opt_bound:

        print(f"Optimize {cfg.bound.type} bound")   
        if cfg.bound.stochastic:
            print("Evaluate bound regularizations over mini-batch")
            bound = lambda n, m, r: BOUNDS[cfg.bound.type](n, m, r, cfg.bound.delta, coeff=2**cfg.training.risk)

        else:
            print("Evaluate bound regularizations over whole train+val set")
            bound = lambda n, m, r: BOUNDS[cfg.bound.type](len(data.X_train) + len(data.X_valid), m, r, cfg.bound.delta, coeff=2**cfg.training.risk)

    trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)
    valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers, shuffle=False)
    testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers, shuffle=False)

    trainvalloader = DataLoader(TorchDataset(np.vstack([data.X_train, data.X_valid]), np.vstack([data.y_train, data.y_valid])), batch_size=cfg.training.batch_size*2, num_workers=cfg.num_workers)

    train_errors, test_errors, bounds, times, risks = [], [], [], [], []
    for i in range(cfg.num_trials):
        
        deterministic(cfg.training.seed+i)

        SAVE_DIR = ROOT_DIR / f"seed={cfg.training.seed+i}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        model = MajorityVote(predictors, beta, mc_draws=cfg.training.MC_draws, distr="categorical")

        monitor = MonitorMV(SAVE_DIR, normalize=True)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)
        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

        *_, best_train_stats, test_error, train_error, time = stochastic_routine(trainloader, valloader, trainvalloader, testloader, model, optimizer, lr_scheduler, bound, cfg.bound.type, loss=loss, loss_eval=loss, monitor=monitor, num_epochs=cfg.training.num_epochs)

        train_errors.append(train_error['error'])
        test_errors.append(test_error['error'])
        bounds.append(best_train_stats[cfg.bound.type])
        risks.append(best_train_stats['error'])
        times.append(time)
        
        monitor.close()
        
    np.save(ROOT_DIR / "err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "train-risk": (np.mean(risks), np.std(risks)), "time": (np.mean(times), np.std(times))})

if __name__ == "__main__":
    main()
