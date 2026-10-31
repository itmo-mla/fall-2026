from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )

    grid = np.column_stack([xx.ravel(), yy.ravel()])

    scores = model.decision_function(grid)
    scores = scores.reshape(xx.shape)
    regions = (scores >= 0).astype(int)

    plt.figure(figsize=(8, 6))
    plt.contourf(
        xx,
        yy,
        regions,
        levels=[-0.5, 0.5, 1.5],
        alpha=0.2,
    )

    plt.contour(
        xx,
        yy,
        scores,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="black",
    )

    if hasattr(model, "support_vectors_"):
        plt.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors="none",
            edgecolors="black",
            s=120,
            linewidths=1.5,
        )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
