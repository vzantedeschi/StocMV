import torch
import numpy as np

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

    return stumps, np.prod(signs.shape)