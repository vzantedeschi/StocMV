import hydra
from time import time
from pathlib import Path

import numpy as np
import torch

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from graphics.plot_predictions import plot_2D
import matplotlib.pyplot as plt

from core.utils import deterministic
from data.datasets import Dataset
from models.naive_bayes import NaiveBayes
from models.stumps import uniform_decision_stumps
from models.random_forest import decision_trees, trees_predict

@hydra.main(config_path='config/non_PB.yaml')
def main(cfg):

    SAVE_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset.name}/{cfg.model.type}/M={cfg.model.M}/max-depth={cfg.model.tree_depth}/seeds={cfg.training.seed}-{cfg.training.seed+cfg.num_trials}/"
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("results will be saved in:", SAVE_DIR.resolve()) 
    
    if cfg.model.tree_depth == "None":
        cfg.model.tree_depth = None 

    models = {
        "GaussianNB": (GaussianNB, {}),
        "Adaboost": (AdaBoostClassifier, {"base_estimator": DecisionTreeClassifier(max_depth=cfg.model.tree_depth), "n_estimators": cfg.model.M}),
        "FrequentistNB": (NaiveBayes, {"frequentist": True}),
        "BayesianNB": (NaiveBayes, {"frequentist": False}),
    }

    if cfg.dataset.name in ["normals", "moons"]:
        load_params = {"n_train": cfg.dataset.N_train, "n_test": cfg.dataset.N_test}
        plot = True
    else:
        load_params = {"normalize": True, "data_path": Path(hydra.utils.get_original_cwd()) / "data"}
        plot = False

    train_errors, test_errors = [], []
    for i in range(cfg.num_trials):
        
        deterministic(cfg.training.seed+i)

        data = Dataset(cfg.dataset.name, **load_params) 

        if cfg.model.type in ["FrequentistNB", "BayesianNB"]:

            if cfg.model.tree_depth == 1:
                predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

            else:

                if cfg.model.tree_depth == "None":
                    cfg.model.tree_depth = None

                trees, M = decision_trees(cfg.model.M, (data.X_train, data.y_train), max_samples=1., max_features="sqrt", max_depth=cfg.model.tree_depth)
                predictors = lambda x: trees_predict(x, trees, binary=data.binary)

            model = models[cfg.model.type][0](voters=predictors, **models[cfg.model.type][1])

            train_x, train_y, test_x, test_y = torch.from_numpy(data.X_train).float(), torch.from_numpy(data.y_train).unsqueeze(1).float(), torch.from_numpy(data.X_test).float(), torch.from_numpy(data.y_test).unsqueeze(1).float()

        else:
            model = models[cfg.model.type][0](**models[cfg.model.type][1])
            M = cfg.model.M

            train_x, train_y = data.X_train, data.y_train

        model.fit(train_x, train_y)

        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        test_error = (test_y[:, 0] != test_pred).sum() / len(test_x)
        train_error = (train_y[:, 0] != train_pred).sum() / len(train_x)

        test_errors.append(test_error)
        train_errors.append(train_error)

        if plot:
            plot_2D(data, model)

            plt.title(f"stumps, M={cfg.model.M}")

            plt.savefig(SAVE_DIR / f"n={cfg.dataset.N_train}-M={M}.pdf", bbox_inches='tight', transparent=True)
            plt.clf()

    np.save(SAVE_DIR / f"err-b.npy", {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors))})


if __name__ == "__main__":
    main()
