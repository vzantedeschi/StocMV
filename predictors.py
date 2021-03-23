import numpy as np

# support only binary classification
def stumps_predict(x, thresholds, signs):
    
    return signs * (1 - 2*(x[..., None] > thresholds).reshape((len(x), -1)))

def uniform_decision_stumps(M, d, min_v, max_v):

    thresholds = np.linspace(min_v, max_v, M, endpoint=False, axis=-1) # get M evenly spaced thresholds in the interval [min_v, max_v] per dimension
    
    sigs = np.ones(M * d * 2) 
    sigs[M * d:] = -1 # first M stumps return one class, last M return the other

    stumps = lambda x: stumps_predict(x, np.hstack((thresholds, thresholds)), sigs)

    return stumps, len(sigs)