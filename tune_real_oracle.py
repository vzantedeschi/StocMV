import optuna
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

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/optuna/{cfg.dataset}/{cfg.training.risk}/{cfg.bound.type}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/M={cfg.model.M}/seed={cfg.training.seed}/"

    ROOT_DIR = Path(ROOT_DIR)
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print("results will be saved in:", ROOT_DIR.resolve()) 

    data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data")

    loss = lambda x, y, z: moment_loss(x, y, z, order=cfg.training.risk)

    bound = None
    if cfg.training.opt_bound:

        print(f"Optimize {cfg.bound.type} bound")   
        if cfg.bound.stochastic:
            print("Evaluate bound over mini-batch")
            bound = lambda n, m, r: BOUNDS[cfg.bound.type](n, m, r, cfg.bound.delta, coeff=2**cfg.training.risk)

        else:
            print("Evaluate bound over whole train+val set")
            bound = lambda n, m, r: BOUNDS[cfg.bound.type](len(data.X_train) + len(data.X_valid), m, r, cfg.bound.delta, coeff=2**cfg.training.risk)

    deterministic(cfg.training.seed)

    predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))
    
    def objective(trial):

        LR = trial.suggest_loguniform('LR', 1e-4, 1.)
        BATCH = 2**trial.suggest_int('BATCH', 6, 11)

        trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH, num_workers=cfg.num_workers, shuffle=True)
        valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH*2, num_workers=cfg.num_workers, shuffle=False)

        trainvalloader = DataLoader(TorchDataset(np.vstack([data.X_train, data.X_valid]), np.vstack([data.y_train, data.y_valid])), batch_size=BATCH*2, num_workers=cfg.num_workers)

        beta = torch.ones(M) / M # prior
        
        model = MajorityVote(predictors, beta, mc_draws=cfg.training.MC_draws, distr="categorical")

        optimizer = Adam(model.parameters(), lr=LR)
        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
        try:
            _, best_val_bound, *_ = stochastic_routine(trainloader, valloader, trainvalloader, None, model, optimizer, lr_scheduler, bound, cfg.bound.type, loss=loss, loss_eval=loss, num_epochs=cfg.training.num_epochs)
        except:
            best_val_bound = 1.
        return best_val_bound

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=cfg.dataset)
    study.optimize(objective, n_trials=20)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(ROOT_DIR / 'trials.csv')

if __name__ == "__main__":
    main()
