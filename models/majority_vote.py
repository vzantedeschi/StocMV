import numpy as np
import torch

from core.distributions import distr_dict

class MajorityVote(torch.nn.Module):

    def __init__(self, voters, prior, mc_draws=10, distr="dirichlet", kl_factor=1.):

        super(MajorityVote, self).__init__()
        
        if distr not in ["dirichlet", "categorical"]:
            raise NotImplementedError

        assert all(prior > 0), "all prior parameters must be positive"
        self.num_voters = len(prior)

        self.prior = prior
        self.voters = voters
        self.mc_draws = mc_draws
        self.distr_type = distr
        post = torch.rand(self.num_voters) * 2 + 1e-9 # uniform draws in (0, 2]
        self.post = torch.nn.Parameter(torch.log(post), requires_grad=True)  # use log (and apply exp(post) later so that posterior parameters are always positive)
        self.distribution = distr_dict[distr](self.post, mc_draws)
        self.kl_factor = kl_factor

    def forward(self, x):
        return self.voters(x)

    def risk(self, batch, loss=None, mean=True):

        if loss is not None:
            return self.distribution.approximated_risk(batch, loss, mean)

        return self.distribution.risk(batch, mean)

    def voter_strength(self, data):
        """ expected accuracy of a voter of the ensemble"""
        # import pdb; pdb.set_trace()
        
        y_target, y_pred = data

        l = torch.where(y_target == y_pred, torch.tensor(1.), torch.tensor(0.))

        return l.mean(1)

    def predict(self, X):
        
        thetas = self.distribution.rsample()
        y_pred = self(X).transpose(1, 0).float()

        labels = thetas @ y_pred

        # TODO: prediction for multiclass but not one-hot encoded
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

        return self.kl_factor * self.distribution.KL(self.prior)

    def get_post(self):
        return torch.exp(self.post)

    def get_post_grad(self):
        return self.post.grad

    def set_post(self, value):

        assert all(value > 0), "all posterior parameters must be positive"
        assert len(value) == self.num_voters
         
        if self.distr_type == "categorical": # make sure params sum to 1
            value /= value.sum()

        self.post = torch.nn.Parameter(torch.log(value), requires_grad=True) # use log (and apply exp(post) later so that posterior parameters are always positive)

        self.distribution.alpha = self.post

    def entropy(self):
        return self.distribution.entropy()

class MultipleMajorityVote(torch.nn.Module):

    def __init__(self, voter_sets, priors, weights, mc_draws=10, posteriors=None, distr="dirichlet",  kl_factor=1.):

        super(MultipleMajorityVote, self).__init__()

        assert len(voter_sets) == len(priors), "must specify same number of voter_sets and priors"
        assert sum(weights) == 1., weights

        if posteriors is not None:
            assert len(priors) == len(posteriors), "must specify same number of priors and posteriors"

            self.mvs = torch.nn.ModuleList([MajorityVote(voters, prior, mc_draws=mc_draws, posterior=post, distr=distr, kl_factor=kl_factor) for voters, prior, post in zip(voter_sets, priors, posteriors)])

        else:
            self.mvs = torch.nn.ModuleList([MajorityVote(voters, prior, mc_draws=mc_draws, distr=distr, kl_factor=kl_factor) for voters, prior in zip(voter_sets, priors)])

        self.weights = weights

    def forward(self, xs):

        return [mv(x) for mv, x in zip(self.mvs, xs)]

    def risk(self, batchs, loss=None, mean=True):

        risks = []
        for mv, w, batch in zip(self.mvs, self.weights, batchs):
            # import pdb; pdb.set_trace()
            risks.append(w * mv.risk(batch, loss, mean))

        return sum(risks)

    def voter_strength(self, batchs):
        """ expected accuracy of a voter of the ensemble"""
        l = torch.stack([w * sum(mv.voter_strength(batch)) for mv, w, batch in zip(self.mvs, self.weights, batchs)])

        return l

    def predict(self, X):
        
        return sum([w * mv.predict(X) for mv, w in zip(self.mvs, self.weights)])

    def KL(self):

        return sum([w * mv.KL() for mv, w in zip(self.mvs, self.weights)])

    def get_post(self):
        return torch.cat([mv.post for mv in self.mvs], 0)

    def get_post_grad(self):
        return torch.cat([mv.post.grad for mv in self.mvs], 0)

    def set_post(self, value):

        for mv in self.mvs:
            mv.set_post(value)

    def entropy(self):

        return sum([w * mv.entropy() for mv, w in zip(self.mvs, self.weights)])
