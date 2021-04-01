import copy
import math
import torch
import numpy as np


class MajorityVoteDiff(torch.nn.Module):

    def __init__(self, majority_vote, device):
        super().__init__()

        self.mv = majority_vote
        self.quasi_uniform = self.mv.quasi_uniform
        self.device = device

        self.prior = torch.tensor(self.mv.prior, device=self.device)
        self._post = torch.nn.Parameter(
            torch.tensor(self.mv.prior, device=self.device))

    def forward(self, batch):
        assert isinstance(self.prior, torch.Tensor)
        assert self.mv.prior.shape == tuple(self.prior.shape)

        x = batch["x"]
        x = x.view(x.shape[0], -1)

        out = self.mv.output(x).double()

        if(not(self.quasi_uniform)):
            #  assert self.mv.complemented
            self.post = torch.nn.functional.softmax(self._post, dim=0)
        else:
            assert not(self.mv.complemented)
            self.post = torch.sigmoid(self._post)
            self.post = (2.0*self.post*(1.0/len(self.mv.voter_list))
                         - 1.0/len(self.mv.voter_list))

        pred = (out@self.post)

        self.out = out
        self.pred = pred
        # pred -> (size, 1)
        assert (len(self.pred.shape) == 2
                and self.pred.shape[0] == x.shape[0]
                and self.pred.shape[1] == 1)
        # out -> (size, nb_voter)
        assert (len(self.out.shape) == 2
                and self.out.shape[0] == x.shape[0]
                and self.out.shape[1] == self.post.shape[0])

        if(self.quasi_uniform):
            self.kl = torch.tensor(0.0, device=self.post.device)
        else:
            self.kl = torch.sum(self.post*torch.log(self.post/self.prior))

    def switch_complemented(self):
        assert self.quasi_uniform
        n = len(self.mv.voter_list)

        if(self.mv.complemented):
            post_n = self._post[:n//2]
            post_2n = self._post[n//2:]
            post = post_n-post_2n
            self._post.data = post
            self.mv.switch_complemented()
            self.prior = torch.tensor(self.mv.prior, device=self.prior.device)
        else:
            post_ = 0.5*(self._post+1.0/(n))
            self._post.data = torch.cat((post_, (1.0/n)-post_), axis=0)
            self.mv.switch_complemented()
            self.prior = torch.tensor(self.mv.prior, device=self.prior.device)
