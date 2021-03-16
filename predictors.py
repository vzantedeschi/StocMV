from sklearn.metrics import accuracy_score

# support only binary classification
class StumpClassifier():

    def __init__(self, index, threshold, sign):

        self.id = index
        self.thr = threshold
        self.sign = sign

    def predict(self, x):
        return self.sign * (1 - 2*(x[self.id] > self.thr))

    def score(self, x, y, sample_weight=None):

        y_pred = self.predict(x)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

def decision_stumps(M, d, min_v, max_v):

    per_d = (M + d - 1) // d
    # get nb_clfs/dimensions regular thresholds
    interval = (max_v - min_v) / per_d
    thresholds = [min_v + (i+1)*interval for i in range(per_d)]

    base_clfs = []

    for j in range(d): 
        base_clfs += [StumpClassifier(j, t, 1) for t in thresholds]
        base_clfs += [StumpClassifier(j, t, -1) for t in thresholds]

    return base_clfs