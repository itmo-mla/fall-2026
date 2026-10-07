"""Выполнение шагов 1 и 2 лабораторной работы № 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_versicolor_virginica
from margins import centroid_linear_rule, geometric_margin


def plot_margins(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    class_names: tuple[str, str],
    weights: np.ndarray,
    bias: float,
    margins: np.ndarray,
    output_path: Path,
) -> None:
    """Показать выборку, границу решения и распределение отступов."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    class_colors = {-1.0: "tab:blue", 1.0: "tab:orange"}

    for label, name in zip((-1.0, 1.0), class_names, strict=True):
        mask = labels == label
        axes[0].scatter(
            features[mask, 0],
            features[mask, 1],
            color=class_colors[label],
            label=name,
            edgecolor="black",
            alpha=0.8,
        )

    x_padding = 0.2
    y_padding = 0.15
    x_limits = (
        features[:, 0].min() - x_padding,
        features[:, 0].max() + x_padding,
    )
    y_limits = (
        features[:, 1].min() - y_padding,
        features[:, 1].max() + y_padding,
    )
    x_line = np.linspace(*x_limits)
    y_line = -(weights[0] * x_line + bias) / weights[1]
    axes[0].plot(x_line, y_line, "k--", label="Базовая линейная граница")
    error_mask = margins <= 0
    axes[0].scatter(
        features[error_mask, 0],
        features[error_mask, 1],
        s=130,
        facecolors="none",
        edgecolors="red",
        linewidths=1.8,
        label="Неположительный отступ",
    )
    axes[0].set_title("Iris: два пересекающихся класса")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].set_xlim(x_limits)
    axes[0].set_ylim(y_limits)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    bins = np.linspace(margins.min(), margins.max(), 18)
    for label, name in zip((-1.0, 1.0), class_names, strict=True):
        axes[1].hist(
            margins[labels == label],
            bins=bins,
            alpha=0.65,
            color=class_colors[label],
            label=name,
            edgecolor="black",
        )
    axes[1].axvline(0.0, color="red", linestyle="--", label="M = 0")
    axes[1].set_title("Распределение геометрических отступов")
    axes[1].set_xlabel("Геометрический отступ")
    axes[1].set_ylabel("Количество объектов")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    dataset = load_iris_versicolor_virginica()
    weights, bias = centroid_linear_rule(dataset.features, dataset.labels)
    margins = geometric_margin(
        dataset.features, dataset.labels, weights, bias
    )

    lab_directory = Path(__file__).resolve().parents[1]
    plot_path = lab_directory / "figures" / "iris_margins.png"
    plot_margins(
        dataset.features,
        dataset.labels,
        dataset.feature_names,
        dataset.class_names,
        weights,
        bias,
        margins,
        plot_path,
    )

    mistakes = int(np.count_nonzero(margins <= 0.0))
    print(f"Объектов: {len(dataset.labels)}")
    print(f"Классы: {dataset.class_names[0]} (-1), {dataset.class_names[1]} (+1)")
    print(f"Признаки: {', '.join(dataset.feature_names)}")
    print(f"Веса базовой границы: {weights}")
    print(f"Смещение базовой границы: {bias:.6f}")
    print(f"Минимальный отступ: {margins.min():.6f}")
    print(f"Средний отступ: {margins.mean():.6f}")
    print(f"Объектов с M <= 0: {mistakes} из {len(margins)}")
    print(f"График сохранён: {plot_path}")


if __name__ == "__main__":
    main()
