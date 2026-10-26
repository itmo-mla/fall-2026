import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results"
)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _sv_indices(model):
    """Возвращает индексы опорных векторов для нашего SVM или sklearn."""
    if hasattr(model, "sv_mask") and model.sv_mask is not None:
        return np.where(model.sv_mask)[0]
    if hasattr(model, "support_"):
        return model.support_
    return None


def _plot_boundary(ax,
                   predict_func, X, y, title,
                   sv_idx=None, grid_size=400):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-1.5, 0, 1.5],
                alpha=0.2, colors=["#ffaaaa", "#aaaaff"])
    ax.contour(xx, yy, Z, levels=[0], colors="k")

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")

    if sv_idx is not None and len(sv_idx) > 0:
        ax.scatter(X[sv_idx, 0], X[sv_idx, 1],
                   s=100, facecolors="none", edgecolors="g",
                   linewidths=1.5, label="SV")
        ax.legend()

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)


def plot_comparison(models, names, X, y,
                    title_suffix="", fname="comparison.png",
                    save_dir=FIGURES_DIR):
    _ensure_dir(save_dir)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, model, name in zip(axes, models, names):
        _plot_boundary(ax, model.predict, X, y,
                       f"{name}{title_suffix}",
                       _sv_indices(model))

    fig.tight_layout()
    out_path = os.path.join(save_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[fig] сохранено: {out_path}")
    return out_path