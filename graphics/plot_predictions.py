import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_2D(data, model, res=0.02, margin=0.1):

    # create a mesh to plot in (points spread uniformly over the space)
    H = .02  # step size in the mesh
    x1_min, x1_max = data.X_train[:,0].min() - margin, data.X_train[:,0].max() + margin
    x2_min, x2_max = data.X_train[:,1].min() - margin, data.X_train[:,1].max() + margin

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res)) # test points

    # estimate learned class boundaries
    test_x = np.c_[xx.ravel(), yy.ravel()]

    y_pred = model.predict(torch.from_numpy(test_x))
    y_pred = torch.argmax(y_pred, 1).detach().numpy()
    y_pred = y_pred.reshape(xx.shape)

    # plot leaf boundaries
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.tab20c, alpha=0.6)

    # plot training points with true labels
    plt.scatter(data.X_train[data.y_train[:, 0] == -1][:,0], data.X_train[data.y_train[:, 0] == -1][:,1], s=20, marker="o", c='k')
    plt.scatter(data.X_train[data.y_train[:, 0] == 1][:,0], data.X_train[data.y_train[:, 0] == 1][:,1], s=20, marker="^", c='k')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())