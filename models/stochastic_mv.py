import numpy as np
import torch

from core.distributions import distr_dict


class MajorityVote(torch.nn.Module):

    def __init__(self, voters, prior, posterior=None, distr="dirichlet"):

        super(MajorityVote, self).__init__()

        assert all(prior > 0), "all prior parameters must be positive"

        self.prior = prior
        self.voters = voters

        if posterior is not None:
            assert all(posterior > 0.), "all posterior parameters must be positive"
            self.post = torch.log(posterior)
        else:
            self.post = torch.nn.Parameter(torch.log(torch.rand(prior.shape) * 2 + 1e-9), requires_grad=True)
        
        self.distribution = distr_dict[distr](self.post)

        self.fitted = False

    def forward(self, x):
        return self.voters(x)

    def risk(self, batch, loss=None):

        if loss is not None:
            return self.distribution.approximated_risk(batch, loss)

        return self.distribution.risk(batch)

    def predict(self, X, num_draws=10):
        
        # thetas = self.rsample(num_draws)
        # y_pred = self(X)

        # return torch.argmax()
        pass

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

    return stumps, d * M * 2