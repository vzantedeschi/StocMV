import torch
import math


def kl_inv(q, epsilon, mode, tol=0.001, nb_iter_max=1000):
    assert mode == "MIN" or mode == "MAX"
    assert isinstance(q, float) and q >= 0 and q <= 1
    assert isinstance(epsilon, float) and epsilon > 0.0

    def kl(q, p):
        return q*math.log(q/p)+(1-q)*math.log((1-q)/(1-p))

    if(mode == "MAX"):
        p_max = 1.0
        p_min = q
    else:
        p_max = q
        p_min = 10.0**-9

    for _ in range(nb_iter_max):
        p = (p_min+p_max)/2.0

        if(kl(q, p) == epsilon or (p_max-p_min)/2.0 < 10**-9):
            return p

        if(mode == "MAX" and kl(q, p) > epsilon):
            p_max = p
        elif(mode == "MAX" and kl(q, p) < epsilon):
            p_min = p
        elif(mode == "MIN" and kl(q, p) > epsilon):
            p_min = p
        elif(mode == "MIN" and kl(q, p) < epsilon):
            p_max = p

    return p


class klInvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, epsilon, mode):
        assert mode == "MIN" or mode == "MAX"
        assert isinstance(q, torch.Tensor) and len(q.shape) == 0
        assert (isinstance(epsilon, torch.Tensor)
                and len(epsilon.shape) == 0 and epsilon > 0.0)
        ctx.save_for_backward(q, epsilon)
        out = kl_inv(q.item(), epsilon.item(), mode)

        if(out < 0.0):
            out = 10.0**-9

        out = torch.tensor(out, device=q.device)
        ctx.out = out
        ctx.mode = mode
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, epsilon = ctx.saved_tensors
        grad_q = None
        grad_epsilon = None

        term_1 = (1.0-q)/(1.0-ctx.out)
        term_2 = (q/ctx.out)

        grad_q = torch.log(term_1/term_2)/(term_1-term_2)
        grad_epsilon = (1.0)/(term_1-term_2)

        return grad_output*grad_q, grad_output*grad_epsilon, None
