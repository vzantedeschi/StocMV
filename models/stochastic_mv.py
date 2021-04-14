import numpy as np
import torch

from core.distributions import distr_dict

class MajorityVote(torch.nn.Module):

    def __init__(self, voters, prior, mc_draws=10, posterior=None, distr="dirichlet"):

        super(MajorityVote, self).__init__()
        
        if distr not in ["dirichlet", "categorical"]:
            raise NotImplementedError

        assert all(prior > 0), "all prior parameters must be positive"

        self.prior = prior
        self.voters = voters
        self.mc_draws = mc_draws

        if posterior is not None:
            assert all(posterior > 0.), "all posterior parameters must be positive"
            self.post = posterior
        else:
            self.post = torch.rand(prior.shape) * 2 + 1e-9 # uniform draws in (0, 2]

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
        return self.distribution.KL(self.prior)

# ------------------------------------------------------------------------------- STUMPS
# support only binary classification
def stumps_predict(x, thresholds, signs):

    return (signs * (1 - 2*(x[..., None] > thresholds))).reshape((len(x), -1))

def uniform_decision_stumps(M, d, min_v, max_v):

    thresholds = torch.from_numpy(np.linspace(min_v, max_v, M, endpoint=False, axis=-1)).float() # get M evenly spaced thresholds in the interval [min_v, max_v] per dimension

    sigs = torch.ones((d, M * 2))
    sigs[..., M:] = -1 # first M*d stumps return one class, last M*d return the other

    stumps = lambda x: stumps_predict(x, torch.cat((thresholds, thresholds), 1), sigs)

    return stumps, d * M * 2

def custom_decision_stumps(thresholds, signs):
    assert thresholds.shape == signs.shape, "have to specify one threshold-sign pair per stump"

    stumps = lambda x: stumps_predict(x, thresholds, signs)

    return stumps, np.prod(signs.shape)