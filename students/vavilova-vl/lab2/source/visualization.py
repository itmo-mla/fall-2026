from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from data_preparation import LABEL_NAMES


COLORS = {0: "#2878B5", 1: "#E07A2D"}
MODEL_COLORS = ["#176B5B", "#2878B5", "#7A5AA6", "#8C8C8C", "#E07A2D", "#C2413B"]


def plot_loo_risk(risks: dict[int, float], best_k: int, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ks = list(risks)
    axis.plot(ks, [risks[k] for k in ks], marker="o", markersize=4, color="#176B5B")
    axis.scatter(
        best_k, risks[best_k], color="#C2413B", s=70, zorder=3,
        label=f"Лучшее k={best_k}, риск={risks[best_k]:.3f}",
    )
    axis.set(title="Эмпирический риск по LOO", xlabel="Число соседей k", ylabel="Доля ошибок")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_compactness_profile(
    profile_full: np.ndarray,
    profile_references: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
) -> None:
    figure, (profile_axis, weight_axis) = plt.subplots(
        1, 2, figsize=(12, 4.8), constrained_layout=True
    )
    m = np.arange(1, len(profile_full) + 1)
    profile_axis.plot(m, profile_full, marker="o", markersize=4, color="#8C8C8C",
                      label="Вся обучающая выборка")
    profile_axis.plot(m, profile_references, marker="o", markersize=4, color="#E07A2D",
                      label="Соседи из эталонов Ω")
    profile_axis.set(
        title="Профиль компактности Π(m)",
        xlabel="Номер соседа m",
        ylabel="Доля объектов, у которых m-й сосед из другого класса",
    )
    profile_axis.grid(alpha=0.25)
    profile_axis.legend()

    weight_axis.bar(np.arange(1, len(weights) + 1), weights, color="#176B5B")
    weight_axis.set_yscale("log")
    weight_axis.set(
        title="Веса R(m) в формуле CCV = Σ Π(m) R(m)",
        xlabel="Номер соседа m",
        ylabel="R(m), логарифмическая шкала",
    )
    weight_axis.grid(alpha=0.25, axis="y")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_selection_history(
    ccv_history: list[float],
    test_errors: list[float],
    start_size: int,
    full_ccv: float,
    full_test_error: float,
    output_path: Path,
) -> None:
    sizes = np.arange(start_size, start_size + len(ccv_history))
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(sizes, ccv_history, color="#C2413B", linewidth=2, label="CCV(Ω) на обучении")
    axis.plot(sizes, test_errors, color="black", linewidth=1, label="Доля ошибок 1NN(Ω) на тесте")
    axis.axhline(full_ccv, color="#C2413B", linestyle="--", linewidth=1,
                 label=f"CCV на всей выборке ({full_ccv:.3f})")
    axis.axhline(full_test_error, color="black", linestyle=":", linewidth=1,
                 label=f"1NN на всей выборке, тест ({full_test_error:.3f})")
    axis.set(
        title="Жадное добавление эталонов",
        xlabel="Число эталонов |Ω|",
        ylabel="Доля ошибок",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_references(
    X: np.ndarray,
    y: np.ndarray,
    reference_indices: np.ndarray,
    noise_indices: np.ndarray,
    output_path: Path,
) -> None:
    pca = PCA(n_components=2).fit(X)
    projected = pca.transform(X)
    is_reference = np.zeros(len(y), dtype=bool)
    is_reference[reference_indices] = True
    is_noise = np.zeros(len(y), dtype=bool)
    is_noise[noise_indices] = True

    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for label in np.unique(y):
        class_mask = y == label
        name = LABEL_NAMES[int(label)]
        color = COLORS[int(label)]
        axis.scatter(*projected[class_mask & ~is_reference & ~is_noise].T,
                     color=color, alpha=0.25, s=22, label=f"{name}: неинформативные")
        axis.scatter(*projected[class_mask & is_noise].T,
                     color=color, marker="^", s=36, alpha=0.8, label=f"{name}: шумовые")
        axis.scatter(*projected[class_mask & is_reference].T,
                     color=color, edgecolor="black", marker="*", s=160,
                     label=f"{name}: эталоны")
    explained = pca.explained_variance_ratio_
    axis.set(
        title="Эталоны, отобранные по CCV (проекция PCA)",
        xlabel=f"PC1 ({explained[0]:.0%} дисперсии)",
        ylabel=f"PC2 ({explained[1]:.0%} дисперсии)",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_model_comparison(results: pd.DataFrame, output_path: Path) -> None:
    labels = results["model"]
    colors = MODEL_COLORS[: len(results)]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    quality_axis = axes[0]
    positions = np.arange(len(results))
    width = 0.38
    for offset, metric, hatch in [(-width / 2, "accuracy", ""), (width / 2, "f1", "//")]:
        bars = quality_axis.bar(positions + offset, results[metric], width,
                                color=colors, hatch=hatch, edgecolor="white", label=metric)
        quality_axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    quality_axis.set_xticks(positions, labels, rotation=20, ha="right")
    quality_axis.set_ylim(0, 1.05)
    quality_axis.set_ylabel("Значение на тестовой выборке")
    quality_axis.set_title("Качество классификации (сплошные — accuracy, штрих — F1)")
    quality_axis.grid(axis="y", alpha=0.25)

    size_axis = axes[1]
    size_bars = size_axis.bar(positions, results["train_objects"], color=colors)
    size_axis.bar_label(size_bars, padding=3)
    size_axis.set_xticks(positions, labels, rotation=20, ha="right")
    size_axis.set_ylabel("Число объектов в обучающей базе")
    size_axis.set_title("Размер обучающей базы")
    size_axis.grid(axis="y", alpha=0.25)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    labels = sorted(LABEL_NAMES)
    columns = 3
    rows = int(np.ceil(len(predictions) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 4 * rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, (model_name, model_predictions) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, model_predictions, labels=labels)
        axis.imshow(matrix, cmap="YlGnBu")
        threshold = matrix.max() / 2
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                          color="white" if matrix[row, column] > threshold else "black")
        axis.set(
            title=model_name,
            xlabel="Предсказанный класс",
            ylabel="Истинный класс",
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=[LABEL_NAMES[label] for label in labels],
            yticklabels=[LABEL_NAMES[label] for label in labels],
        )
    for axis in axes[len(predictions):]:
        axis.axis("off")
    figure.suptitle("Матрицы ошибок на тестовой выборке")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
