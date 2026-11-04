import matplotlib.pyplot as plt
import numpy as np


def plot_boundary(ax, model, X, y, title):
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
    )
    f = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, f, levels=[-np.inf, 0, np.inf], colors=["#cfe3f5", "#f9d6c9"], alpha=0.7)
    ax.contour(xx, yy, f, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])

    for label, color, name in ((-1, "tab:blue", "класс -1"), (1, "tab:red", "класс +1")):
        ax.scatter(*X[y == label].T, c=color, s=22, alpha=0.8, label=name)
    ax.scatter(*model.X_[model.boundary_].T, s=110, facecolors="none", edgecolors="k",
               linewidths=1.5, label=f"опорные граничные ({model.boundary_.sum()})")
    ax.scatter(*model.X_[model.violators_].T, s=110, facecolors="none", edgecolors="limegreen",
               linewidths=1.5, marker="s", label=f"опорные нарушители ({model.violators_.sum()})")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("признак 1 (стандартизованный)")
    ax.set_ylabel("признак 2 (стандартизованный)")
    ax.legend(fontsize=7, loc="lower left")


def plot_models(models, X, y, filename, ncols=3, suptitle=None):
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
    for ax, (title, m) in zip(axes.ravel(), models):
        plot_boundary(ax, m, X, y, title)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(str(filename), dpi=200)
    plt.close(fig)


def plot_margins(model, X, y, filename, title):
    m = model.margins(X, y)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(m, bins=30, color="tab:purple", alpha=0.8)
    ax.axvline(0, color="k", linewidth=2, label="M = 0 (граница)")
    ax.axvline(1, color="gray", linestyle="--", label="M = 1 (край полосы)")
    ax.set_xlabel("отступ M_i = y_i * f(x_i)")
    ax.set_ylabel("число объектов")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(filename), dpi=200)
    plt.close(fig)


def plot_grid(scores, Cs, gammas, filename, title):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(scores, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(gammas)), [f"{g:g}" for g in gammas])
    ax.set_yticks(range(len(Cs)), [f"{c:g}" for c in Cs])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title(title)
    for i in range(len(Cs)):
        for j in range(len(gammas)):
            ax.text(j, i, f"{scores[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, label="accuracy (CV)")
    fig.tight_layout()
    fig.savefig(str(filename), dpi=200)
    plt.close(fig)


def plot_quality(results, filename):
    names = list(results)
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(metrics)
    for k, met in enumerate(metrics):
        vals = [results[n][met] for n in names]
        ax.bar(np.arange(len(names)) + k * width, vals, width, label=met)
    ax.set_xticks(np.arange(len(names)) + width * (len(metrics) - 1) / 2, names, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("значение метрики на тесте")
    ax.set_title("Качество: собственный DualSVM и sklearn.SVC")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(filename), dpi=200)
    plt.close(fig)
