import torch

from sklearn.ensemble import RandomForestClassifier

def trees_predict(x, trees, binary=True):
    
    pred = torch.stack([torch.from_numpy(t.predict(x)).float() for t in trees], 1)

    if binary:
        pred[pred == 0] = -1

    return pred

def decision_trees(M, data, max_samples=1., max_features="sqrt", max_depth=None):

    bootstrap = True
    if max_samples == 1.:
        bootstrap = False
        max_samples = None

    forest = RandomForestClassifier(n_estimators=M, criterion="gini", max_depth=max_depth, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples, n_jobs=-1, class_weight='balanced_subsample')
    forest.fit(*data)

    return forest.estimators_, M

def two_forests(M, r, X, y, max_samples, max_depth, binary):

    assert 0 <= r <= 1, r
    
    m = int(len(X) * r)

    # learn one prior
    trees1, _ = decision_trees(M, (X[:m], y[:m]), max_samples=max_samples, max_depth=max_depth)

    # learn the other prior
    trees2, _ = decision_trees(M, (X[m:], y[m:]), max_samples=max_samples, max_depth=max_depth)

    predictors1 = lambda x: trees_predict(x, trees1, binary=binary)
    predictors2 = lambda x: trees_predict(x, trees2, binary=binary)

    return (predictors1, predictors2), M