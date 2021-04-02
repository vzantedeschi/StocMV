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
            monitor.write_all(i, torch.exp(model.post), train={"Train-obj": cost.item()})

def train_stochastic(dataloader, model, optimizer, epoch, bound=None, loss=None, monitor=None):

    model.train()

    last_iter = epoch * len(dataloader)
    train_obj = 0.

    pbar = tqdm(dataloader)

    for i, batch in enumerate(pbar):

        # import pdb; pdb.set_trace()
        optimizer.zero_grad()

        risk = model.risk(batch, loss)

        if bound is not None:
            cost = bound(len(batch[0]), model, risk)
        else:
            cost = risk
            
        train_obj += cost.item()

        if prog_bar:
            pbar.set_description("avg train obj %f" % (train_obj / (i + 1)))

        cost.backward()

        optimizer.step()
        
        if monitor:
            monitor.write_all(i, torch.exp(model.post), train={"Train-obj": cost.item()})
            
def evaluate(dataloader, model, bounds, epoch, monitor=None, contrastive=False, device=None):

    model.eval()

    total_metrics = {k: 0. for k in bounds.keys()}

    risk = 0.
    for batch in dataloader:

        risk += model.risk(batch).item()

    total_metrics.update({"error": risk})

    for k in bounds.keys():
        b = bounds[k](len(batch[0]), model, risk)
        total_metrics[k] += b.item()

    if monitor:
        monitor.write(epoch, val=total_metrics)

    return total_metrics