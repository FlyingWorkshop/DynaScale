import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA

MAX_LINES = 30


def _plot2d(trajs_grid, gridlabels: list = None, ax : plt.Axes = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    for i, trajs in enumerate(trajs_grid):
        if gridlabels is None:
            line_collection = LineCollection(trajs, color=f"C{i}")
        else:
            line_collection = LineCollection(
                trajs, color=f"C{i}", label=gridlabels[i])
        ax.add_collection(line_collection)
        ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1], color=f"C{i}", marker="o")
    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    if gridlabels is not None:
        ax.legend()
    return fig, ax


def _plot3d(grid, gridlabels: list = None, ax : plt.Axes = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()
        if ax.name != "3d":
            #replace the ax with a 3d projection in the same position
            position = ax.get_position()
            fig.delaxes(ax)
            ax = fig.add_subplot(position, projection='3d')
    for i, trajs in enumerate(grid):
        if gridlabels is None:
            line_collection = Line3DCollection(trajs, color=f"C{i}")
        else:
            line_collection = Line3DCollection(
                trajs, color=f"C{i}", label=gridlabels[i])
        ax.add_collection(line_collection)
        ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1],
                   trajs[:, 0, 2], color=f"C{i}", marker="o")
    minima = grid.min(axis=(0, 1, 2))
    maxima = grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    ax.set_zlim(minima[2], maxima[2])
    if gridlabels is not None:
        ax.legend()
    return fig, ax


def _plot4d(trajs_grid, pca: PCA, gridlabels: list = None, ax: plt.Axes = None):
    assert pca.n_components == 2 or pca.n_components == 3
    trajs_grid = _apply_pca_to_grid(trajs_grid, pca)
    if pca.n_components == 2:
        return _plot2d(trajs_grid, gridlabels=gridlabels, ax=ax)
    else:
        return _plot3d(trajs_grid, gridlabels=gridlabels, ax=ax)


def make_pca(trajs: np.ndarray, n_components=3):
    dim = trajs.shape[-1]
    X = trajs.reshape((-1, dim))
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def _apply_pca_to_grid(trajs_grid, pca) -> np.ndarray:
    dim = trajs_grid.shape[-1]
    X = trajs_grid.reshape((-1, dim))
    result = pca.transform(X)
    target_shape = list(trajs_grid.shape)
    target_shape[-1] = pca.n_components
    result = result.reshape(target_shape)
    return result


def plot(
        datasets: list[np.ndarray], 
        target_dim: int = 3, 
        max_lines=MAX_LINES, 
        specieslabels: list[str] = None, 
        labels: list[str] = None,
        title: str = None
    ):
    datasets = np.array([x[:max_lines] for x in datasets])
    dim = datasets.shape[-1]
    fig = plt.figure(figsize=(16, 4))
    posidx = 1
    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(1, len(datasets)+1, posidx)

        for i in range(len(dataset[0][0])):
            ydata = dataset[0][:, i]
            if specieslabels:
                ax.plot(ydata, label=specieslabels[i], alpha=0.8)
            else:
                ax.plot(ydata, label=f"Species {i}", alpha=0.8)
            ax.set_title(labels[idx])
            ax.legend()
        posidx += 1

    assert target_dim <= dim
    assert target_dim <= 3
    ax =  fig.add_subplot(1, len(datasets)+1, posidx)
    if dim == 2:
        fig, ax = _plot2d(datasets, gridlabels=labels, ax=ax)
    elif dim == 3 and target_dim == 3:
        fig, ax = _plot3d(datasets, gridlabels=labels, ax=ax)
    else:
        pca = make_pca(datasets, n_components=target_dim)
        fig, ax =  _plot4d(datasets,  pca, gridlabels=labels, ax=ax)
    if title:
        fig.suptitle(title)
    return fig, ax