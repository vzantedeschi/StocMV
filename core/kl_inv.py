import torch

def kl(q, p):
    return q * torch.log(q/p) + (1-q) * torch.log((1-q)/(1-p))

def kl_inv(q, epsilon, mode, nb_iter_max=1000):
    # bisection optimization method
    
    assert mode in ["MIN", "MAX"]
    assert epsilon >= 0
    assert 0. <= q < 1., q

    if(mode == "MAX"):
        p_max, p_min = 1., q
    else:
        p_max, p_min = q, 1e-9

    for _ in range(nb_iter_max):

        p = (p_min+p_max)/2.
        p_kl = kl(q, p)

        if torch.isclose(p_kl, epsilon) or (p_max-p_min)/2. < 1e-9:
            return p

        if mode == "MAX":
            
            if p_kl > epsilon:
                p_max = p

            elif p_kl < epsilon:
                p_min = p

        elif p_kl > epsilon:
            p_min = p

        elif p_kl < epsilon:
            p_max = p

    return p


class klInvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, epsilon, mode="MAX"):

        ctx.save_for_backward(q, epsilon)
        out = kl_inv(q, epsilon, mode)
        out = torch.clamp(out, 1e-9, 1-1e-4)

        ctx.out = out
        ctx.mode = mode

        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, epsilon = ctx.saved_tensors

        term_1 = (1. - q)/(1. - ctx.out)
        term_2 = q / ctx.out

        grad_q = torch.log(term_1/term_2) / (term_1-term_2)
        grad_epsilon = 1. / (term_1-term_2)

        return grad_output * grad_q, grad_output * grad_epsilon, None
