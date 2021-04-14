import hydra
import optuna
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.bounds import BOUNDS
from core.losses import sigmoid_loss
from core.monitors import MonitorMV
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.stochastic_mv import MajorityVote, uniform_decision_stumps

from training_routines import stochastic_routine

@hydra.main(config_path='config/real.yaml')
def main(cfg):

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/optuna/{cfg.dataset}/{cfg.training.risk}/{cfg.bound.type}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/M={cfg.model.M}/prior={cfg.model.prior}/seed={cfg.training.seed}/"
    ROOT_DIR = Path(ROOT_DIR)
    ROOT_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", ROOT_DIR.resolve()) 

    data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data")

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

    deterministic(cfg.training.seed)

    predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))
    
    def objective(trial):

        LR = trial.suggest_loguniform('LR', 1e-6, 1)
        BATCH = 2**trial.suggest_int('BATCH', 6, 11)

        trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH, num_workers=cfg.num_workers, shuffle=True)
        valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH*2, num_workers=cfg.num_workers, shuffle=False)

        trainvalloader = DataLoader(TorchDataset(np.vstack([data.X_train, data.X_valid]), np.vstack([data.y_train, data.y_valid])), batch_size=BATCH*2, num_workers=cfg.num_workers)

        beta = torch.ones(M) * cfg.model.prior # prior
        
        model = MajorityVote(predictors, beta, mc_draws=cfg.training.MC_draws)

        optimizer = Adam(model.parameters(), lr=LR)
        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

        _, best_val_bound, *_ = stochastic_routine(trainloader, valloader, trainvalloader, None, model, optimizer, lr_scheduler, bound, cfg.bound.type, loss=loss, num_epochs=cfg.training.num_epochs)
        
        return best_val_bound

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=cfg.dataset)
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(ROOT_DIR / 'trials.csv')

if __name__ == "__main__":
    main()
