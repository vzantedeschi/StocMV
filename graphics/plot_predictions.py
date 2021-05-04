import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac"]
markers = ["o", "^"]
cmap = ListedColormap(colors, name="custom")

def plot_2D(data, model, bound=None, res=0.02, margin=0.1, classes=[-1, 1]):

    # create a mesh to plot in (points spread uniformly over the space)
    H = .02  # step size in the mesh
    x1_min, x1_max = data.X_train[:,0].min() - margin, data.X_train[:,0].max() + margin
    x2_min, x2_max = data.X_train[:,1].min() - margin, data.X_train[:,1].max() + margin

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res)) # test points

    # estimate learned class boundaries
    test_x = np.c_[xx.ravel(), yy.ravel()]

    try: # torch model
        y_pred = model.predict(torch.from_numpy(test_x))

        if y_pred.dim() == 2:
            y_pred = torch.argmax(y_pred, 1)

        y_pred = y_pred.detach().numpy()

    except: # numpy model
        y_pred = model.predict(test_x)

    y_pred = y_pred.reshape(xx.shape)

    # plot leaf boundaries
    plt.contourf(xx, yy, y_pred, cmap=cmap, alpha=0.6)

    for m, c in zip(markers, classes):
        # plot training points with true labels
        plt.scatter(data.X_train[data.y_train == c][:,0], data.X_train[data.y_train == c][:,1], s=20, marker=m, c="w", edgecolors="k")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    if bound:
        plt.text(xx.max() - .1, yy.min() + .3, ('%.2f' % bound).lstrip('0'), size=30, horizontalalignment='right')