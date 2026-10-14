from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_split, standardize
from prototype_selection import select_prototypes


def main():
    x_train_raw, x_test_raw, y_train, _, feature_names, class_names = load_iris_split()
    x_train, _, _, _ = standardize(x_train_raw, x_test_raw)
    result = select_prototypes(x_train, y_train, k=1, noise_threshold=0.0)

    colors = ("tab:blue", "tab:orange", "tab:green")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for class_index, class_name, color in zip(
        range(len(class_names)), class_names, colors, strict=True
    ):
        class_mask = y_train == class_index
        axes[0].scatter(
            x_train_raw[class_mask, 2],
            x_train_raw[class_mask, 3],
            color=color,
            alpha=0.35,
            label=class_name,
        )

    axes[0].scatter(
        x_train_raw[result.prototype_indices, 2],
        x_train_raw[result.prototype_indices, 3],
        marker="*",
        s=220,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Эталоны",
    )
    axes[0].scatter(
        x_train_raw[result.noise_indices, 2],
        x_train_raw[result.noise_indices, 3],
        marker="x",
        s=100,
        color="red",
        linewidths=2,
        label="Шум",
    )
    axes[0].set_title("Результат отбора эталонов")
    axes[0].set_xlabel(feature_names[2])
    axes[0].set_ylabel(feature_names[3])
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    iterations = np.arange(len(result.error_history))
    prototype_counts = 3 + iterations
    axes[1].plot(
        prototype_counts,
        result.error_history,
        marker="o",
        label="Ошибки на очищенном train",
    )
    axes[1].set_title("Добавление эталонов")
    axes[1].set_xlabel("Количество эталонов")
    axes[1].set_ylabel("Количество ошибок")
    axes[1].set_xticks(prototype_counts)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "prototypes.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Эталонов: {len(result.prototype_indices)}")
    print(f"Шумовых объектов: {len(result.noise_indices)}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
