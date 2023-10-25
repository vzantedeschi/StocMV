from pathlib import Path

import numpy as np
import torch

from torch import lgamma, digamma
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.distributions import Dirichlet
from core.kl_inv import klInvFunction
from core.losses import sigmoid_loss
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.stumps import uniform_decision_stumps

from optimization import stochastic_routine

class MajorityVote(torch.nn.Module):

    def __init__(self, voters, prior, mc_draws=10):

        super(MajorityVote, self).__init__()

        assert all(prior > 0), "all prior parameters must be positive"
        self.num_voters = len(prior)

        self.prior = prior
        self.voters = voters # callable
        self.mc_draws = mc_draws

        post = torch.rand(self.num_voters) * 2 + 1e-9 # uniform draws in (0, 2]
        self.post = torch.nn.Parameter(torch.log(post), requires_grad=True)  # use log (and apply exp(post) later so that posterior parameters are always positive)
        
        self.distribution = Dirichlet(self.post, mc_draws)

    def forward(self, x):
        return self.voters(x) # must return a tensor of predictions of dimensions [batch_size, num_voters]

    def risk(self, batch, loss=None, mean=True):

        if loss is not None: # for MC bound
            return self.distribution.approximated_risk(batch, loss, mean)

        # for exact bound
        return self.distribution.risk(batch, mean)

    def KL(self):
        return self.distribution.KL(self.prior)

    def get_post(self):
        return torch.exp(self.post)

    def get_post_grad(self):
        return self.post.grad

    def set_post(self, value):

        assert all(value > 0), "all posterior parameters must be positive"
        assert len(value) == self.num_voters

        self.post = torch.nn.Parameter(torch.log(value), requires_grad=True) # use log (and apply exp(post) later so that posterior parameters are always positive)

        self.distribution.alpha = self.post

    def entropy(self):
        return self.distribution.entropy()

    def voter_strength(self, data):
        """ expected accuracy of a voter of the ensemble"""
        
        y_target, y_pred = data

        l = torch.where(y_target == y_pred, torch.tensor(1.), torch.tensor(0.))

        return l.mean(1)

def seeger_bound(n, model, risk, delta, verbose=False):

    kl = model.KL()

    const = np.log(2 * (n**0.5) / delta)

    bound = klInvFunction.apply(risk, (kl + const) / n)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, n={n}")
        print(f"Bound={bound.item()}\n")

    return bound 


def main(risk_type):

    num_trials = 2
    seed = 42
    dataset = "TTT"
    m = 100 # number of stumps per class and per axis
    bound_type = "seeger"
    delta = 0.05
    prior = 1
    batch_size = 1024
    num_workers = 4
    MC_draws = 10
    lr = 0.1
    epochs = 100
    
    # define params for each method
    losses = { # type: (loss, distribution-type)
        "exact": None, # exact bound
        "MC": lambda x, y, z: sigmoid_loss(x, y, z, c=100), # MC bound
    }

    train_errors, test_errors, train_losses, bounds, kls, times = [], [], [], [], [], []
    for i in range(num_trials):
        
        print("seed", seed+i)
        deterministic(seed+i)

        # load dataset
        data = Dataset(dataset, normalize=True, valid_size=0)
        
        # prepare stumps. M = 2*m total stumps
        predictors, M = uniform_decision_stumps(m, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0))

        print(f"Optimize {bound_type} bound")

        print("Evaluate bound regularizations over whole training set")
        n = len(data.X_train)

        # callable to compute bound
        bound = lambda _, model, risk: seeger_bound(n, model, risk, delta=delta)

        trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=batch_size, num_workers=num_workers, shuffle=True)
        testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096, num_workers=num_workers, shuffle=False)

        betas = torch.ones(M) * prior # prior

        # majority vote
        model = MajorityVote(predictors, betas, MC_draws)

        optimizer = Adam(model.parameters(), lr=lr) 
        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

        *_, best_train_stats, train_error, test_error, time = stochastic_routine(trainloader, testloader, model, optimizer, bound, bound_type, loss=losses[risk_type], num_epochs=epochs, lr_scheduler=lr_scheduler)

        train_errors.append(train_error['error'])
        test_errors.append(test_error['error'])
        kls.append(model.KL().detach().numpy())
        bounds.append(best_train_stats[bound_type])
        times.append(time)

        if risk_type == "MC":
            train_losses.append(best_train_stats["error"]) # available only for non-exact methods

    results = {"train-error": (np.mean(train_errors), np.std(train_errors)), "test-error": (np.mean(test_errors), np.std(test_errors)), bound_type: (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times)), "train-risk": (np.mean(train_losses), np.std(train_losses)), "KL": (np.mean(kls), np.std(kls))}
    print(results)

if __name__ == "__main__":
    import sys

    risk_type = sys.argv[1] # MC or exact
    main(risk_type=risk_type)
