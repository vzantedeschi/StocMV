import hydra
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset

from core.bounds import BOUNDS
from core.losses import sigmoid_loss, moment_loss, rand_loss
from core.monitors import MonitorMV
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MultipleMajorityVote, MajorityVote
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps

from optimization import stochastic_routine

@hydra.main(config_path='config/real.yaml')
def main(cfg):

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/M={cfg.model.M}/max-depth={cfg.model.tree_depth}/prior={cfg.model.prior}/"

    ROOT_DIR = Path(ROOT_DIR)

    if cfg.model.uniform:
        ROOT_DIR /= "uniform"
    else:
        ROOT_DIR /= f"lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/"

    if cfg.training.risk == "MC":
        ROOT_DIR /= f"MC={cfg.training.MC_draws}"

    print("results will be saved in:", ROOT_DIR.resolve()) 

    # define params for each method
    risks = { # type: (loss, bound-coeff, distribution-type, kl factor)
        "exact": (None, 1., "dirichlet", 1.),
        "MC": (lambda x, y, z: sigmoid_loss(x, y, z, c=cfg.training.sigmoid_c), 1., "dirichlet", 1.),
        "Rnd": (lambda x, y, z: rand_loss(x, y, z, n=cfg.training.rand_n), 2., "categorical", cfg.training.rand_n),
        "FO": (lambda x, y, z: moment_loss(x, y, z, order=1), 2., "categorical", 1.),
        "SO": (lambda x, y, z: moment_loss(x, y, z, order=2), 4., "categorical", 2.),
    }

    train_errors, test_errors, train_losses, bounds, strengths, entropies, kls, times = [], [], [], [], [], [], [], []
    for i in range(cfg.num_trials):
        
        print("seed", cfg.training.seed+i)
        deterministic(cfg.training.seed+i)

        SAVE_DIR = ROOT_DIR / f"seed={cfg.training.seed+i}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if (SAVE_DIR / "err-b.npy").is_file():
            print(SAVE_DIR)
            # load saved stats
            seed_results = np.load(SAVE_DIR / "err-b.npy", allow_pickle=True).item()

        else:

            seed_results = {}

            data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data", valid_size=0)

            m = 0
            if cfg.model.pred == "stumps-uniform":
                predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

            elif cfg.model.pred == "rf": # random forest

                if cfg.model.tree_depth == "None":
                    cfg.model.tree_depth = None

                predictors, M = two_forests(cfg.model.M, 0.5, data.X_train, data.y_train, max_samples=cfg.model.bootstrap, max_depth=cfg.model.tree_depth, binary=data.binary)

            else:
                raise NotImplementedError("model.pred should be one the following: [stumps-uniform, rf]")

            loss, coeff, distr, kl_factor = risks[cfg.training.risk]

            bound = None
            if cfg.training.opt_bound:

                print(f"Optimize {cfg.bound.type} bound")

                if cfg.bound.stochastic:
                    print("Evaluate bound regularizations over mini-batch")
                    bound = lambda n, model, risk: BOUNDS[cfg.bound.type](n, model, risk, delta=cfg.bound.delta, coeff=coeff)

                else:
                    print("Evaluate bound regularizations over whole training set")
                    n = len(data.X_train)
                    bound = lambda _, model, risk: BOUNDS[cfg.bound.type](n, model, risk, delta=cfg.bound.delta, coeff=coeff)

            if cfg.model.pred == "rf": # a loader per posterior

                m_train = len(data.X_train) // 2
                train1 = TorchDataset(data.X_train[m_train:], data.y_train[m_train:])
                train2 = TorchDataset(data.X_train[:m_train], data.y_train[:m_train])
                trainloader = [
                    DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                    DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                ] 

            else:
                train = TorchDataset(data.X_train, data.y_train)
                trainloader = DataLoader(train, batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)

            testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096, num_workers=cfg.num_workers, shuffle=False)

            if cfg.model.pred == "rf":
                betas = [torch.ones(M) * cfg.model.prior for p in predictors] # prior

                # weights proportional to data sizes
                model = MultipleMajorityVote(predictors, betas, weights=(0.5, 0.5), mc_draws=cfg.training.MC_draws, distr=distr, kl_factor=kl_factor)
            
            else:
                betas = torch.ones(M) * cfg.model.prior # prior

                model = MajorityVote(predictors, betas, mc_draws=cfg.training.MC_draws, distr=distr, kl_factor=kl_factor)

            monitor = MonitorMV(SAVE_DIR)
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)
            # init learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

            *_, best_train_stats, train_error, test_error, time = stochastic_routine(trainloader, testloader, model, optimizer, bound, cfg.bound.type, loss=loss, monitor=monitor, num_epochs=cfg.training.num_epochs, lr_scheduler=lr_scheduler)
    
        
            seed_results["train-error"] = train_error['error']
            seed_results["test-error"] = test_error['error']
            seed_results["train-risk"] = best_train_stats["error"]
            seed_results[cfg.bound.type] = best_train_stats[cfg.bound.type]
            seed_results["time"] = time
            seed_results["posterior"] = model.get_post().detach().numpy()
            seed_results["strength"] = best_train_stats["strength"]
            seed_results["KL"] = model.KL().item()
            seed_results["entropy"] = model.entropy().item()

            # save seed results
            np.save(SAVE_DIR / "err-b.npy", seed_results)
            monitor.close()

        train_errors.append(seed_results["train-error"])
        test_errors.append(seed_results["test-error"])
        entropies.append(seed_results["entropy"])
        strengths.append(seed_results["strength"])
        kls.append(seed_results["KL"])
        bounds.append(seed_results[cfg.bound.type])
        times.append(seed_results["time"])
        train_losses.append(seed_results.pop("train-risk", None)) # available only for non-exact methods
 
    results = {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times)), "strength": (np.mean(strengths), np.std(strengths)), "train-risk": (np.mean(train_losses), np.std(train_losses)), "entropy": (np.mean(entropies), np.std(entropies)), "KL": (np.mean(kls), np.std(kls))}

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()
