import torch
from torch.optim import Optimizer

###############################################################################


class COCOB(Optimizer):

    def __init__(self, params, weight_decay=0, alpha=100):
        defaults = dict(weight_decay=weight_decay)
        super(COCOB, self).__init__(params, defaults)

        assert weight_decay >= 0.0
        assert alpha > 0.0
        self.weight_decay = weight_decay
        self.alpha = alpha

        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # We update all the parameters with Algorithm 2 (COCOB-Backprop)
        # that was introduced in [1]
        for group in self.param_groups:
            for w in group['params']:

                # We get the gradient
                grad = w.grad

                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "COCOB does not support sparse gradients")

                # We initalize the state
                if(w not in self.state):
                    self.state[w] = {}
                state = self.state[w]

                # We initialize the initial weights
                if("w_1" not in state):
                    state["w_1"] = w.data.clone()

                # We add the weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(w, alpha=group['weight_decay'])

                # We get the negative gradient (line 4)
                grad = -grad

                # We update the maximum observed scale (line 6)
                if("L" not in state):
                    state["L"] = torch.tensor(1.0e-9, device=grad.device)
                state["L"] = torch.max(state["L"], torch.abs(grad))

                # We update the sum of the absolute values of the grad (line 7)
                if("G" not in state):
                    state["G"] = torch.tensor(0.0, device=grad.device)
                state["G"] = state["G"] + torch.abs(grad)

                # We update the sum of the grad (line 9)
                if("theta" not in state):
                    state["theta"] = torch.tensor(0.0, device=grad.device)
                state["theta"] = state["theta"] + grad

                # We update the reward (line 8)
                if("reward" not in state):
                    state["reward"] = torch.tensor(0.0, device=grad.device)
                state["reward"] = (state["reward"]
                                   + (w.data - state["w_1"])*grad)
                state["reward"] = torch.max(
                    torch.tensor(0.0, device=grad.device), state["reward"])

                # We compute the associated wealth
                wealth = state["reward"] + state["L"]

                # We compute the beta (line 10)
                beta = torch.max(state["G"]+state["L"], self.alpha*state["L"])
                beta = state["theta"]/(state["L"]*beta)

                # We calculate the parameters (line 10)
                w.data = state["w_1"] + beta*wealth

        return loss

###############################################################################

# References:
# [1] Training Deep Networks without Learning Rates Through Coin Betting
#     Francesco Orabona, Tatiana Tommasi, 2017
