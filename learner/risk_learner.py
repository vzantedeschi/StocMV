import torch
import math
from learner.gradient_descent_learner import GradientDescentLearner
from core.cocob_optim import COCOB

###############################################################################

class RiskLearner(GradientDescentLearner):

    def __init__(self, majority_vote, epoch=1, batch_size=None):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size)
        self._optim = None

    def _optimize(self, batch):
        if(self._optim is None):
            self._optim = COCOB(self.mv_diff.parameters())

        self.mv_diff(batch)
        pred = self.mv_diff.pred

        assert "y" in batch and isinstance(batch["y"], torch.Tensor)
        y = batch["y"]
        y_unique = torch.sort(torch.unique(y))[0]
        assert y_unique[0].item() == -1 and y_unique[1].item() == +1

        assert len(y.shape) == 2 and len(pred.shape) == 2
        assert (pred.shape[0] == y.shape[0] and pred.shape[1] == y.shape[1]
                and y.shape[1] == 1)

        self._loss = 0.5*(1.0-torch.mean(y*pred))

        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

###############################################################################
