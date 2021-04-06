import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def load_normals(n_train, n_test, means, scales):

    train_x, train_y = [], []
    test_x, test_y = [], []

    for i, (m, s) in enumerate(zip(means, scales)):
        train_x.append(np.random.multivariate_normal(m, s, n_train))
        test_x.append(np.random.multivariate_normal(m, s, n_test))

        train_y.append(i * np.ones(n_train))
        test_y.append(i * np.ones(n_test))

    train_x, train_y, test_x, test_y = np.vstack(train_x), np.concatenate(train_y), np.vstack(test_x), np.concatenate(test_y)

    if len(means) == 2:
        train_y[train_y == 0] = -1
        test_y[test_y == 0] = -1

    return train_x, train_y, test_x, test_y
    
def load_moons(n_train, n_test, noise=0.05, normalize=True):

    train_x, train_y = datasets.make_moons(n_samples=n_train, noise=noise)
    test_x, test_y = datasets.make_moons(n_samples=n_test, noise=noise)

    train_y[train_y == 0] = -1
    test_y[test_y == 0] = -1

    if normalize:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    return train_x, train_y, test_x, test_y

DATASETS = {
    "normals": load_normals,
    "moons": load_moons
}

def toy_dataset(name, n_train, n_test, **kwargs):

    if name == "normals":
        X_train, y_train, X_test, y_test = DATASETS[name](n_train, n_test, means=((-1, 0), (1, 0)), scales=(np.diag([0.1, 1]), np.diag([0.1, 1])))

    else:
        X_train, y_train, X_test, y_test = DATASETS[name](n_train, n_test)
        
    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)