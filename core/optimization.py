import torch

from tqdm import tqdm

def train_batch(data, model, optimizer, learner=None, bound=None, loss=None, nb_iter=1e4, monitor=None):

    model.train()

    n = len(data[0])

    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:
        optimizer.zero_grad()

        if bound is not None:
            cost = bound(n, model, model.risk(data, loss))

        elif learner is not None:
            cost = learner.loss(n, model, data)

        else:
            cost = model.risk(data, loss)

        pbar.set_description("train obj %s" % cost.item())

        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write_all(i, torch.exp(model.post), model.post.grad, train={"Train-obj": cost.item()})

def train_stochastic(dataloader, model, optimizer, epoch, learner=None, bound=None, loss=None, monitor=None):

    model.train()

    last_iter = epoch * len(dataloader)
    train_obj = 0.

    pbar = tqdm(dataloader)

    for i, batch in enumerate(pbar):
        # import pdb; pdb.set_trace()

        n = len(batch[0])
        data = batch[1], model(batch[0])

        # import pdb; pdb.set_trace()
        optimizer.zero_grad()

        if bound is not None:
            cost = bound(n, model, model.risk(data, loss))

        elif learner is not None:
            cost = learner.loss(n, model, data)

        else:
            cost = model.risk(data, loss)
            
        train_obj += cost.item()

        pbar.set_description("avg train obj %f" % (train_obj / (i + 1)))

        cost.backward()
        optimizer.step()
        
        if monitor:
            monitor.write_all(last_iter+i, torch.exp(model.post), model.post.grad, train={"Train-obj": cost.item()})
            
def evaluate(dataloader, model, epoch, bounds=None, loss=None, monitor=None, tag="val"):

    model.eval()

    risk = 0.
    n = 0
    for n_b, batch in enumerate(dataloader):

        data = batch[1], model(batch[0])

        risk += model.risk(data, loss=loss, mean=False)
        n += len(data[0])

    risk /= n
    total_metrics = {"error": risk.item()}

    if bounds is not None:

        for k in bounds.keys():
            total_metrics[k] = bounds[k](n, model, risk).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics