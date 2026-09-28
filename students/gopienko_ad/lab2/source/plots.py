from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True,)
    return path


def plot_loo_risk(
        k_values: np.ndarray,
        risks: np.ndarray,
        best_k: int,
        path: str | Path
) -> None:

    path = _ensure_parent(path)
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        risks,
        marker="o",
    )
    plt.axvline(
        best_k,
        linestyle="--",
        label=f"best k = {best_k}",
    )

    plt.xlabel("k")
    plt.ylabel("LOO empirical risk")
    plt.title("LOO risk for variable-width Parzen KNN")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160,)
    plt.close()


def plot_metric_comparison(
    metrics_by_model: dict[str, dict[str, float]],
    path: str | Path
) -> None:
    path = _ensure_parent(path)

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    model_names = list(metrics_by_model.keys())

    x = np.arange(len(metric_names))
    width = 0.8 / max(len(model_names), 1)

    plt.figure(figsize=(10, 5))

    for model_index, model_name in enumerate(model_names):
        values = [metrics_by_model[model_name][metric]
            for metric in metric_names
        ]

        offset = (model_index - (len(model_names) - 1) / 2) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=model_name,
        )

    plt.xticks(x, metric_names)
    plt.ylim(0.0, 1.05,)
    plt.ylabel("score")
    plt.title("Model quality comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def pca_2d(x: np.ndarray) -> np.ndarray:

    X = np.asarray(x, dtype=float)
    centered = X - np.mean(X, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]

    return centered @ components.T


def plot_prototypes_2d(
    X: np.ndarray,
    y: np.ndarray,
    prototype_indices: np.ndarray,
    path: str | Path
) -> None:
    path = _ensure_parent(path)

    projection = pca_2d(X)
    prototype_mask = np.zeros(len(X), dtype=bool,)
    prototype_mask[prototype_indices] = True

    plt.figure(figsize=(8, 6))

    for class_label in np.unique(y):
        class_mask = (y == class_label)

        plt.scatter(
            projection[class_mask & ~prototype_mask,0],
            projection[class_mask & ~prototype_mask, 1],
            alpha=0.35,
            s=20,
            label=(
                f"class {class_label}: "
                "non-prototypes"
            )
        )

        plt.scatter(
            projection[class_mask & prototype_mask, 0],
            projection[class_mask & prototype_mask, 1],
            marker="x",
            s=70,
            linewidths=2,
            label=(
                f"class {class_label}: "
                "prototypes"
            )
        )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Prototype selection result")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160,)
    plt.close()


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray],
    path: str | Path
) -> None:
    path = _ensure_parent(path)

    model_names = list(matrices.keys())
    n_models = len(model_names)

    figure, axes = plt.subplots(
        1,
        n_models,
        figsize=(5 * n_models, 4),
        squeeze=False,
    )

    for axis, model_name in zip(axes[0], model_names):
        matrix = np.asarray(matrices[model_name], dtype=int)
        axis.imshow(matrix)

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center"
                )

        axis.set_xticks(np.arange(matrix.shape[1]))
        axis.set_yticks(np.arange(matrix.shape[0]))
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(model_name)

    figure.suptitle("Confusion matrices")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
