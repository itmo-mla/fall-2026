import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

CLASSES = {-1: "#2a78d6", 1: "#eb6834"}
REGIONS = ListedColormap(["#dbe8f8", "#fbe3d7"])
REFERENCE = "#1baf7a"


def save(fig, path, rect=(0, 0, 1, 1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_decision(models, X, y, path, references=None):
    """One panel per model: predicted regions, the boundary f(x) = 0, the margin f(x) = +-1 and support vectors.

    `references` are sklearn models under the same keys, their boundary is drawn on top for comparison.
    """
    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
    ys = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5))
    for ax, (title, model) in zip(axes, models.items()):
        F = model.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, F, levels=[F.min(), 0, F.max()], cmap=REGIONS)
        ax.contour(xx, yy, F, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
        if references:
            F_ref = references[title].decision_function(grid).reshape(xx.shape)
            ax.contour(xx, yy, F_ref, levels=[0], colors=REFERENCE, linestyles=":", linewidths=2.5)
        for label, color in CLASSES.items():
            ax.scatter(*X[y == label].T, color=color, s=25, alpha=0.8)

        bound = model.lam >= model.C - model.eps
        ax.scatter(*model.X_sv[~bound].T, facecolors="none", edgecolors="black", s=110, linewidths=1.5)
        ax.scatter(*model.X_sv[bound].T, color="black", marker="x", s=50, linewidths=1.5)
        ax.set_title(f"{title}\nопорных: {len(model.lam)}, из них нарушителей: {bound.sum()}")
        ax.set_xlabel("признак 1")
        ax.set_ylabel("признак 2")

    legend = [
        Line2D([], [], marker="o", linestyle="", color=CLASSES[-1], label="класс −1"),
        Line2D([], [], marker="o", linestyle="", color=CLASSES[1], label="класс +1"),
        Line2D([], [], color="black", linewidth=2, label="граница f(x) = 0"),
        Line2D([], [], color="black", linewidth=1, linestyle="--", label="полоса f(x) = ±1"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=10,
               label="опорный-граничный, 0 < λ < C"),
        Line2D([], [], marker="x", linestyle="", color="black", markersize=8, label="опорный-нарушитель, λ = C"),
    ]
    if references:
        legend.insert(3, Line2D([], [], color=REFERENCE, linewidth=2.5, linestyle=":", label="граница sklearn SVC"))
    fig.legend(handles=legend, loc="lower center", ncol=len(legend))
    save(fig, path, rect=(0, 0.06, 1, 1))
