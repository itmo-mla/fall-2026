"""Визуализации: PCA через SVD (numpy) + scatter plot эталонов."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import load_raw, make_xy, standardize


def pca_2d(X):
    """PCA до 2D на numpy через SVD. X уже стандартизован."""
    Xc = X - X.mean(axis=0)
    # SVD: Xc = U S V^T, главные оси — столбцы V
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def run_visualization(out_dir: Path,
                      grouping: str = "4",
                      sample_idx=None,
                      title_suffix: str = "all"):
    """
    Рисует PCA-проекцию исходных точек.
    sample_idx: массив индексов эталонов (подсветить их поверх).
    """
    df = load_raw()
    X, y, classes = make_xy(load_raw())
    X_std, _ = standardize(X, X.copy())
    Z = pca_2d(X_std)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c_i, c_name in enumerate(classes):
        m = y == c_i
        ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.5, label=c_name)

    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx)
        ax.scatter(Z[sample_idx, 0], Z[sample_idx, 1],
                   s=120, facecolors="none", edgecolors="k",
                   linewidths=1.4, label="прототипы")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA-проекция UCI Wine ({title_suffix})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"pca_scatter_{title_suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out}")