
# support only binary classification
def stump(x, index, threshold, sign):

    return sign * (1 - 2*(x[index] > threshold))

def uniform_decision_stumps(M, d, min_v, max_v):

    per_d = (M // 2 + d - 1) // d
    # get nb_clfs/dimensions regular thresholds
    interval = (max_v - min_v) / per_d
    thresholds = [min_v + (i+1)*interval for i in range(per_d)]
    print(thresholds)
    base_clfs = []

    for j in range(d): 
        base_clfs += [lambda x: stump(x, j, t, 1) for t in thresholds]
        base_clfs += [lambda x: stump(x, j, t, -1) for t in thresholds]

    return base_clfs