import numpy as np
import torch

from core.distributions import distr_dict

class MajorityVote(torch.nn.Module):

    def __init__(self, voters, prior, mc_draws=10, posterior=None, distr="dirichlet"):

        super(MajorityVote, self).__init__()
        
        if distr not in ["dirichlet", "categorical"]:
            raise NotImplementedError

        if len(prior) == 2:
            assert all(prior[0] >= 0) and all(prior[1] >= 0), "all prior parameters must be nonnegative"
            assert prior[0].shape == prior[1].shape, "two priors must have the same shape"

            self.informed_prior = True
            self.num_voters = len(prior[0])

        else:
            assert all(prior >= 0), "all prior parameters must be nonnegative"
            self.informed_prior = False
            self.num_voters = len(prior)

        self.prior = prior
        self.voters = voters
        self.mc_draws = mc_draws

        if posterior is not None:
            assert all(posterior >= 0.), "all posterior parameters must be nonnegative"
            self.post = posterior

        else:

            self.post = torch.rand(self.num_voters) * 2 + 1e-9 # uniform draws in (0, 2]

        if distr == "categorical": # make sure params sum to 1
            self.post /= self.post.sum()

        self.post = torch.nn.Parameter(torch.log(self.post), requires_grad=True) # use log (and apply exp(post) later so that posterior parameters are always positive)
        
        self.distribution = distr_dict[distr](self.post, mc_draws)

        self.fitted = True

    def forward(self, x):
        return self.voters(x)

    def risk(self, batch, loss=None, mean=True):

        if loss is not None:
            return self.distribution.approximated_risk(batch, loss, mean)

        return self.distribution.risk(batch, mean)

    def predict(self, X):
        
        thetas = self.distribution.rsample()
        y_pred = self(X).transpose(1, 0).float()

        labels = thetas @ y_pred

        if y_pred.dim() == 3:
            num_classes = labels.shape[2]
            c_min, c_max = 0, num_classes - 1
            labels = torch.argmax(labels, 2) # if multiclass

        else:
            num_classes = 2
            c_min, c_max = -1, 1
            labels = torch.sign(labels) # if binary

        pred = torch.stack([torch.histc(l, bins=num_classes, min=c_min, max=c_max) / self.mc_draws for l in labels.T])

        return pred 

    def KL(self):

        if self.informed_prior:
            return self.distribution.KL(self.prior[0]) + self.distribution.KL(self.prior[1])

        return self.distribution.KL(self.prior)