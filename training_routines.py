from time import time
from copy import deepcopy

from core.optimization import train_stochastic, evaluate

def stochastic_routine(trainloader, valloader, trainvalloader, testloader, model, optimizer, lr_scheduler, bound, bound_type, loss=None, loss_eval=None, monitor=None, num_epochs=100):

    best_val_bound = float("inf")
    best_e = -1
    no_improv = 0

    t1 = time()
    for e in range(num_epochs):
        train_stochastic(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor)

        val_stats = evaluate(valloader, model, epoch=e, bounds={bound_type: bound}, loss=loss_eval, monitor=monitor)
        train_stats = evaluate(trainvalloader, model, epoch=e, bounds={bound_type: bound}, loss=loss_eval, monitor=monitor, tag="train-val") # just for monitoring purposes
        print(f"Epoch {e}: {val_stats[bound_type]}\n")
        
        no_improv += 1
        if val_stats[bound_type] < best_val_bound:
            best_val_bound = val_stats[bound_type]
            best_train_stats = train_stats
            best_e = e
            best_model = deepcopy(model)
            no_improv = 0

        # reduce learning rate if needed
        lr_scheduler.step(val_stats[bound_type])

        if no_improv == num_epochs // 4:
            break

    t2 = time()

    res = best_model, best_val_bound, best_train_stats

    if testloader is not None:
        test_error = evaluate(testloader, best_model, epoch=e, tag="test")

        print(f"Test error: {test_error['error']}; {bound_type} bound: {best_train_stats[bound_type]}\n")

        res = (*res, test_error)

    if loss_eval is not None:
        train_error = evaluate(trainvalloader, best_model, epoch=e, tag="train-val")

        res = (*res, train_error, t2-t1)

    return (*res, t2-t1)