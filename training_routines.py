from time import time
from copy import deepcopy

from models.majority_vote import MultipleMajorityVote
from core.optimization import train_stochastic, train_stochastic_multiset, evaluate, evaluate_multiset

def stochastic_routine(trainloader, testloader, model, optimizer, bound, bound_type, loss=None, loss_eval=None, monitor=None, num_epochs=100, lr_scheduler=None):

    best_bound = float("inf")
    best_e = -1
    no_improv = 0
    best_train_stats = None

    if isinstance(model, MultipleMajorityVote): # then expect multiple dataloaders
        train_routine = train_stochastic_multiset
        val_routine = evaluate_multiset
        test_routine = lambda d, *args, **kwargs: evaluate_multiset((d, d), *args, **kwargs)
    else:
        train_routine, val_routine, test_routine = train_stochastic, evaluate, evaluate

    
    t1 = time()
    for e in range(num_epochs):
        train_routine(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor)

        train_stats = val_routine(trainloader, model, epoch=e, bounds={bound_type: bound}, loss=loss_eval, monitor=monitor, tag="train") # just for monitoring purposes
        print(f"Epoch {e}: {train_stats[bound_type]}\n")
        
        no_improv += 1
        if train_stats[bound_type] < best_bound:
            best_bound = train_stats[bound_type]
            best_train_stats = train_stats
            best_e = e
            best_model = deepcopy(model)
            no_improv = 0

        # reduce learning rate if needed
        if lr_scheduler:
            lr_scheduler.step(train_stats[bound_type])

        if no_improv == num_epochs // 4:
            break

    t2 = time()

    res = best_model, best_bound, best_train_stats

    if testloader is not None:
        test_error = test_routine(testloader, best_model, epoch=e, tag="test")

        print(f"Test error: {test_error['error']}; {bound_type} bound: {best_train_stats[bound_type]}\n")

        res = (*res, test_error)

    if loss_eval is not None:
        train_error = val_routine(trainloader, best_model, epoch=e, tag="train")

        res = (*res, train_error)

    return (*res, t2-t1)
