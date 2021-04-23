import hydra
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset

from core.bounds import BOUNDS
from core.losses import sigmoid_loss, moment_loss, exp_loss
from core.monitors import MonitorMV
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MultipleMajorityVote, MajorityVote
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps

from training_routines import stochastic_routine

@hydra.main(config_path='config/real.yaml')
def main(cfg):

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/M={cfg.model.M}/prior={cfg.model.prior}/lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/"

    ROOT_DIR = Path(ROOT_DIR)

    if cfg.training.risk == "MC":
        ROOT_DIR = ROOT_DIR / f"MC={cfg.training.MC_draws}"

    print("results will be saved in:", ROOT_DIR.resolve()) 

    # define params for each method
    risks = { # type: (loss, bound-coeff, distribution-type)
        "exact": (None, 1., "dirichlet"),
        "MC": (sigmoid_loss, 1., "dirichlet"),
        "FO": (lambda x, y, z: moment_loss(x, y, z, order=1), 2., "categorical"),
        "SO": (lambda x, y, z: moment_loss(x, y, z, order=2), 4., "categorical"),
        "exp": (lambda x, y, z: exp_loss(x, y, z, c=cfg.training.risk_c), np.exp(cfg.training.risk_c / 2) - 1, "categorical")
    }

    data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data")     
    
    m = 0
    if cfg.model.pred == "stumps-uniform":
        predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

    elif cfg.model.pred == "rf": # random forest

        if cfg.model.tree_depth == "None":
            cfg.model.tree_depth = None

        m = cfg.model.m # number of points for learning first prior
        predictors, M = two_forests(cfg.model.M, cfg.model.m, data.X_train, data.y_train[:, 0], max_samples=cfg.model.bootstrap, max_depth=cfg.model.tree_depth, binary=True)

    else:
        raise NotImplementedError("model.pred should be one the following: [stumps-uniform, rf]")

    loss, coeff, distr = risks[cfg.training.risk]

    bound = None
    if cfg.training.opt_bound:

        print(f"Optimize {cfg.bound.type} bound")

        if cfg.bound.stochastic:
            print("Evaluate bound regularizations over mini-batch")
            bound = lambda n, model, risk: BOUNDS[cfg.bound.type](n, model, risk, delta=cfg.bound.delta, m=int(m*n), coeff=coeff)

        else:
            print("Evaluate bound regularizations over whole train+val set")
            n = len(data.X_train) + len(data.X_valid)
            bound = lambda _, model, risk: BOUNDS[cfg.bound.type](n, model, risk, delta=cfg.bound.delta, m=int(m*n), coeff=coeff)

    if cfg.model.pred == "rf": # a loader per posterior

        m_train = int(len(data.X_train) * cfg.model.m)
        train1 = TorchDataset(data.X_train[m_train:], data.y_train[m_train:])
        train2 = TorchDataset(data.X_train[:m_train], data.y_train[:m_train])
        trainloader = [
            DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
            DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
        ] 

        m_val = int(len(data.X_valid) * cfg.model.m)
        val1 = TorchDataset(data.X_valid[m_val:], data.y_valid[m_val:])
        val2 = TorchDataset(data.X_valid[:m_val], data.y_valid[:m_val])
        valloader = [
            DataLoader(val1, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
            DataLoader(val2, batch_size=4096, num_workers=cfg.num_workers, shuffle=False)
        ]

        trainvalloader = [
            DataLoader(ConcatDataset([train1, val1]), batch_size=4096, num_workers=cfg.num_workers),
            DataLoader(ConcatDataset([train2, val2]), batch_size=4096, num_workers=cfg.num_workers)
        ]

    else:
        train = TorchDataset(data.X_train, data.y_train)
        trainloader = DataLoader(train, batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)

        val = TorchDataset(data.X_valid, data.y_valid)
        valloader = DataLoader(val, batch_size=4096, num_workers=cfg.num_workers, shuffle=False)

        trainvalloader = DataLoader(ConcatDataset([train, val]), batch_size=4096, num_workers=cfg.num_workers)

    testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096, num_workers=cfg.num_workers, shuffle=False)

    train_errors, test_errors, train_losses, bounds, times = [], [], [], [], []
    for i in range(cfg.num_trials):
        
        deterministic(cfg.training.seed+i)

        SAVE_DIR = ROOT_DIR / f"seed={cfg.training.seed+i}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if cfg.model.pred == "rf":
            betas = [torch.ones(M) * cfg.model.prior for p in predictors] # prior

            # weights proportional to data sizes
            model = MultipleMajorityVote(predictors, betas, weights=(cfg.model.m, 1-cfg.model.m), mc_draws=cfg.training.MC_draws, distr=distr)
        
        else:
            betas = torch.ones(M) * cfg.model.prior # prior

            model = MajorityVote(predictors, betas, mc_draws=cfg.training.MC_draws, distr=distr)

        monitor = MonitorMV(SAVE_DIR)
        optimizer = Adam(model.parameters(), lr=cfg.training.lr)
        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

        if cfg.training.risk in ["exact", "MC"]:
            
            *_, best_train_stats, test_error, time = stochastic_routine(trainloader, valloader, trainvalloader, testloader, model, optimizer, lr_scheduler, bound, cfg.bound.type, loss=loss, monitor=monitor, num_epochs=cfg.training.num_epochs)
            
            train_errors.append(best_train_stats['error'])

        else:
            *_, best_train_stats, test_error, train_error, time = stochastic_routine(trainloader, valloader, trainvalloader, testloader, model, optimizer, lr_scheduler, bound, cfg.bound.type, loss=loss, loss_eval=loss, monitor=monitor, num_epochs=cfg.training.num_epochs)
        
            train_errors.append(train_error['error'])
            train_losses.append(best_train_stats['error'])

        test_errors.append(test_error['error'])
        bounds.append(best_train_stats[cfg.bound.type])
        times.append(time)

        monitor.close()

    results = {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times))}

    if cfg.training.risk not in ["exact", "MC"]:
        results.update({"train-risk": (np.mean(train_losses), np.std(train_losses))})

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()
