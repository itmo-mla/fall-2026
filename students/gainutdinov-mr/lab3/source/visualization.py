"""Визуализация экспериментов лабораторной работы № 3."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from kernels import ProductKernel, SumKernel
from sklearn.metrics import ConfusionMatrixDisplay


def kernel_name(kernel):
    if isinstance(kernel, SumKernel):
        return f"({kernel_name(kernel.left)} + {kernel_name(kernel.right)})"
    if isinstance(kernel, ProductKernel):
        return f"({kernel_name(kernel.left)} * {kernel_name(kernel.right)})"
    return str(kernel)


def plot_decision_slice(models, X_train, X_test, y_test, feature_indices,
                        feature_names, output_path, title=""):
    """Меняем два признака, остальные оставляем равными средним по train.

    Точки показывают реальные классы тестовых объектов. Их остальные признаки
    отличаются от среднего, поэтому класс точки может не совпадать с фоном.
    """
    points = X_test[:, feature_indices]
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.1, 0.5)
    xx, yy = np.meshgrid(
        np.linspace(lower[0] - padding[0], upper[0] + padding[0], 120),
        np.linspace(lower[1] - padding[1], upper[1] + padding[1], 120),
    )
    samples = np.tile(X_train.mean(axis=0), (xx.size, 1))
    samples[:, feature_indices[0]] = xx.ravel()
    samples[:, feature_indices[1]] = yy.ravel()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)
    for ax, (name, model) in zip(axes.flat, models.items()):
        # Пакеты ограничивают размер матрицы ядра при вычислении сетки.
        scores = np.concatenate(
            [model.decision_function(batch) for batch in np.array_split(samples, 24)]
        ).reshape(xx.shape)
        ax.contourf(xx, yy, (scores >= 0).astype(int), levels=[-0.5, 0.5, 1.5],
                    colors=["tab:blue", "tab:orange"], alpha=0.15)
        levels = [level for level in (-1, 0, 1) if scores.min() < level < scores.max()]
        if levels:
            contours = ax.contour(xx, yy, scores, levels=levels, colors="black",
                                  linestyles=["solid" if level == 0 else "dashed"
                                              for level in levels])
            ax.clabel(contours, fmt="%g")
        for label, color in [(-1, "tab:blue"), (1, "tab:orange")]:
            class_points = points[y_test == label]
            ax.scatter(class_points[:, 0], class_points[:, 1], c=color, s=18,
                       label=f"Test class {label:+d}")
        ax.set(title=name, xlabel=f"{feature_names[0]} (standardized)",
               ylabel=f"{feature_names[1]} (standardized)")
        ax.legend()
    if X_train.shape[1] == 2:
        plot_name = "Decision boundary"
    else:
        plot_name = "Decision slice: other features fixed at training means"
    fig.suptitle(f"{plot_name}\n{fill(title, width=100)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics(results, output_path, title=""):
    """Метрики качества на общей шкале, время и число опорных объектов отдельно."""
    names = list(results)
    quality_metrics = ["Accuracy", "Precision", "Recall", "F1"]
    other_metrics = [metric for metric in results[names[0]] if metric not in quality_metrics]
    colors = ["tab:blue", "tab:orange"]
    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 4)
    quality_ax = fig.add_subplot(grid[0, :])
    positions = np.arange(len(quality_metrics))
    width = 0.35
    for index, name in enumerate(names):
        values = [results[name][metric] for metric in quality_metrics]
        bars = quality_ax.bar(positions + (index - 0.5) * width, values,
                              width=width, color=colors[index], label=name)
        quality_ax.bar_label(bars, fmt="%.4f", padding=4)
    quality_ax.set_xticks(positions, quality_metrics)
    quality_values = [results[name][metric] for name in names for metric in quality_metrics]
    min_score = min(quality_values)
    max_score = max(quality_values)
    # Отступ зависит от разброса, но остается и при совпадении всех метрик.
    eps = max((max_score - min_score) * 0.15, 0.01)
    lower_bound = max(0, min_score - eps)
    quality_ax.set_ylim(lower_bound, max_score + 3 * eps)
    quality_ax.set_ylabel("Score (zoomed scale)" if lower_bound > 0 else "Score")
    quality_ax.legend(loc="upper right", ncol=2)
    quality_ax.set_axisbelow(True)
    quality_ax.grid(axis="y", alpha=0.25)

    for index, metric in enumerate(other_metrics):
        ax = fig.add_subplot(grid[1, index])
        values = [results[name][metric] for name in names]
        bars = ax.bar(names, values, color=colors, width=0.55)
        ax.set_ylim(0, max(max(values) * 1.25, 1e-9))
        if metric == "Support vectors":
            number_format = "%.0f"
        elif metric.endswith("(us)"):
            number_format = "%.2f"
        else:
            number_format = "%.6f"
        ax.bar_label(bars, fmt=number_format, padding=4)
        ax.set_title(fill(metric, width=27))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Model comparison\n{fill(title, width=100)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(y_test, predictions, output_path, title=""):
    """Матрицы ошибок на тестовой выборке с одинаковым порядком классов."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5),
                             squeeze=False)
    for ax, (name, y_pred) in zip(axes.flat, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=[-1, 1], display_labels=["-1", "+1"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name)
    fig.suptitle(f"Confusion matrices\n{fill(title, width=100)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

