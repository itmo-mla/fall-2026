import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def plot_solutions(X, y, models, path, eps=1e-6):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 180),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 180),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    colors = ["tab:blue", "tab:orange"]
    regions = ListedColormap(["#dbe8f8", "#fbe3d7"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (title, kernel, lambdas, w0, C) in zip(axes, models):
        support = lambdas > eps
        values = np.dot(kernel(grid, X[support]), lambdas[support] * y[support]) - w0
        values = values.reshape(xx.shape)
        ax.contourf(xx, yy, values >= 0, levels=[-0.5, 0.5, 1.5], cmap=regions)
        ax.contour(
            xx, yy, values, levels=[-1, 0, 1], colors="black",
            linestyles=["--", "-", "--"], linewidths=[1, 1.8, 1],
        )
        for cls, color in zip([-1, 1], colors):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], color=color, s=18, alpha=0.8)

        boundary = support & (lambdas < C - eps)
        bounded = lambdas >= C - eps
        ax.scatter(
            X[boundary, 0], X[boundary, 1], s=85,
            facecolors="none", edgecolors="black", linewidths=1.2,
        )
        ax.scatter(X[bounded, 0], X[bounded, 1], marker="x", color="black", s=35)
        ax.set(
            xlabel="Стандартизированный x1",
            title=f"{title}\nОпорных: {np.sum(support)}",
        )
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Стандартизированный x2")
    legend = [
        Line2D([], [], marker="o", linestyle="", color=colors[0], label="Класс -1"),
        Line2D([], [], marker="o", linestyle="", color=colors[1], label="Класс +1"),
        Line2D([], [], color="black", label="f(x) = 0"),
        Line2D([], [], color="black", linestyle="--", label="f(x) = ±1"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
               markeredgecolor="black", label="0 < λ < C"),
        Line2D([], [], marker="x", linestyle="", color="black", label="λ ≈ C"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
