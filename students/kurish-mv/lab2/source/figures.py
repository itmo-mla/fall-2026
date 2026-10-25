from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from kernel_knn import pairwise_distances
from quality import MODEL_LABELS

MODEL_TITLES = MODEL_LABELS
MODEL_COLORS = {"full": "#335c81", "sklearn": "#4f772d", "reduced": "#8e5572"}
CLASS_COLORS = ("#3d5a80", "#e07a5f", "#81b29a")
METRIC_COLORS = ("#3d5a80", "#98c1d9", "#e07a5f")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_risk_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    best_k: dict[str, int],
    path: Path,
) -> None:
    """Эмпирический риск LOO для собственной и эталонной моделей"""
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    styles = {"full": "-", "sklearn": "--", "reduced": "-."}

    for key, (k_values, risks) in curves.items():
        color = MODEL_COLORS[key]
        minimum = 100.0 * risks.min()
        label = f"{MODEL_TITLES[key]}: k={best_k[key]}, {minimum:.2f}%"
        axis.plot(k_values, 100.0 * risks, styles[key], color=color, linewidth=1.9, label=label)
        position = int(np.flatnonzero(k_values == best_k[key])[0])
        axis.scatter(
            best_k[key], 100.0 * risks[position], s=52, zorder=3,
            facecolors="white", edgecolors=color, linewidths=1.6,
        )
        axis.axvline(best_k[key], color=color, alpha=0.22, linewidth=1.0)

    all_max = max(100.0 * risks.max() for _, risks in curves.values())
    axis.set(
        title="Эмпирический риск LOO при выборе k",
        xlabel="k — номер соседа, задающего ширину окна",
        ylabel="Доля ошибок LOO, %",
        xlim=(0.5, 40.5),
        ylim=(0.0, max(5.0, 1.15 * all_max)),
        xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", frameon=False, fontsize=8.5)
    _save(figure, path)


def plot_pruning_trace(
    history: pd.DataFrame,
    y_train: np.ndarray,
    keep_mask: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    """Ход прореживания и состав оставленных эталонов"""
    figure, (risk_axis, count_axis) = plt.subplots(
        1, 2, figsize=(11.5, 4.8), width_ratios=[1.55, 1.0]
    )

    color = MODEL_COLORS["reduced"]
    risk_axis.step(
        history["prototypes"], 100.0 * history["risk"], where="post",
        color=color, linewidth=2.0,
    )
    risk_axis.fill_between(
        history["prototypes"], 100.0 * history["risk"], step="post",
        color=color, alpha=0.12,
    )
    axis_line = 100.0 * float(history["risk"].iloc[0])
    risk_axis.axhline(axis_line, color="#666666", linestyle="--", linewidth=1.0)
    final_count = int(history["prototypes"].iloc[-1])
    final_risk = 100.0 * float(history["risk"].iloc[-1])
    risk_axis.scatter([final_count], [final_risk], s=55, color=color, zorder=3)
    risk_axis.invert_xaxis()
    risk_axis.set(
        title=f"Риск 1NN: останов на {final_count} эталонах",
        xlabel="Число эталонов",
        ylabel="Ошибка 1NN, %",
    )
    risk_axis.grid(alpha=0.25)

    classes = np.arange(len(class_names))
    original = np.bincount(y_train, minlength=len(classes))
    kept = np.bincount(y_train[keep_mask], minlength=len(classes))
    width = 0.38
    left = count_axis.bar(
        classes - width / 2, original, width, color=CLASS_COLORS[: len(classes)],
        alpha=0.3, edgecolor="#555555", hatch="//", label="исходно",
    )
    right = count_axis.bar(
        classes + width / 2, kept, width, color=CLASS_COLORS[: len(classes)],
        alpha=0.95, label="оставлено",
    )
    count_axis.bar_label(left, padding=2, fontsize=8)
    count_axis.bar_label(right, padding=2, fontsize=8)
    count_axis.set(
        title=f"Состав эталонов: {final_count} из {len(y_train)}",
        xlabel="Класс", ylabel="Число объектов",
        xticks=classes, xticklabels=class_names,
        ylim=(0, 1.18 * original.max()),
    )
    count_axis.grid(axis="y", alpha=0.25)
    count_axis.set_axisbelow(True)
    count_axis.legend(frameon=False, loc="upper right", fontsize=8.5)
    _save(figure, path)


def plot_pca_prototypes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    keep_mask: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    """PCA-проекция с эталонами и связями удалённых объектов"""
    centered = X_train - X_train.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T
    explained = singular[:2] ** 2 / np.sum(singular**2)

    keep = np.flatnonzero(keep_mask)
    dropped = np.flatnonzero(~keep_mask)
    nearest = keep[np.argmin(pairwise_distances(X_train[dropped], X_train[keep]), axis=1)]

    figure, axis = plt.subplots(figsize=(8.6, 6.4))
    for source, target in zip(dropped, nearest):
        axis.plot(
            [coordinates[source, 0], coordinates[target, 0]],
            [coordinates[source, 1], coordinates[target, 1]],
            color="#9a9a9a", alpha=0.16, linewidth=0.6, zorder=0,
        )

    for label, name in enumerate(class_names):
        dropped_mask = ~keep_mask & (y_train == label)
        kept_mask = keep_mask & (y_train == label)
        axis.scatter(
            coordinates[dropped_mask, 0], coordinates[dropped_mask, 1],
            marker="x", s=20, alpha=0.42, linewidths=0.8,
            color=CLASS_COLORS[label], zorder=1,
        )
        axis.scatter(
            coordinates[kept_mask, 0], coordinates[kept_mask, 1],
            marker="o", s=70, facecolors="white", linewidths=1.7,
            edgecolors=CLASS_COLORS[label], zorder=2,
        )

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor="white",
            markeredgecolor=CLASS_COLORS[i], markeredgewidth=1.7, markersize=8,
            label=name,
        )
        for i, name in enumerate(class_names)
    ]
    handles.append(
        Line2D([0], [0], marker="x", linestyle="", color="#888888", markersize=6,
               label="удалённые")
    )
    axis.legend(handles=handles, loc="best", frameon=False, fontsize=8.5)
    axis.text(
        0.015, 0.985, f"эталонов: {len(keep)} из {len(y_train)}",
        transform=axis.transAxes, ha="left", va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8,
              "edgecolor": "#cccccc"},
    )
    axis.set(
        title="PCA-проекция: эталоны и удалённые объекты",
        xlabel=f"Главная компонента 1 ({100 * explained[0]:.1f}%)",
        ylabel=f"Главная компонента 2 ({100 * explained[1]:.1f}%)",
    )
    axis.grid(alpha=0.22)
    _save(figure, path)


def plot_quality_panels(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    class_names: list[str],
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    """Матрицы ошибок трёх моделей и сводные метрики"""
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    keys = ["full", "sklearn", "reduced"]

    for axis, key in zip(axes.ravel()[:3], keys):
        matrix = metrics.loc[key, "confusion"]
        image = axis.imshow(matrix, cmap="Blues")
        for row, column in np.ndindex(matrix.shape):
            value = int(matrix[row, column])
            axis.text(
                column, row, str(value), ha="center", va="center", fontsize=10,
                color="white" if value >= 0.65 * matrix.max() else "#333333",
            )
        axis.set(
            title=(
                f"{MODEL_TITLES[key]}\n"
                f"accuracy = {metrics.loc[key, 'accuracy']:.3f}, "
                f"macro F1 = {metrics.loc[key, 'f1_macro']:.3f}"
            ),
            xlabel="Предсказано", ylabel="Истина",
            xticks=range(len(class_names)), xticklabels=class_names,
            yticks=range(len(class_names)), yticklabels=class_names,
        )
        axis.tick_params(length=0)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    summary = axes[1, 1]
    positions = np.arange(len(keys))
    width = 0.26
    columns = [
        ("loo_risk", "риск LOO", METRIC_COLORS[0]),
        ("accuracy", "accuracy", METRIC_COLORS[1]),
        ("f1_macro", "macro F1", METRIC_COLORS[2]),
    ]
    for offset, (column, label, color) in enumerate(columns):
        values = metrics.loc[keys, column].to_numpy(dtype=float)
        bars = summary.bar(
            positions + (offset - 1) * width, values, width, color=color, label=label
        )
        summary.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    summary.set(
        title="Сводка метрик", ylabel="Значение", ylim=(0.0, 1.15),
        xticks=positions,
        xticklabels=["Парзен", "sklearn", "Парзен\n(эталоны)"],
    )
    summary.grid(axis="y", alpha=0.25)
    summary.set_axisbelow(True)
    summary.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)
    _save(figure, path)
