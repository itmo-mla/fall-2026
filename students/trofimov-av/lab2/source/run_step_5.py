from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from data import load_iris_split, standardize
from evaluation import calculate_metrics
from parzen_knn import ParzenKNN


def draw_confusion_matrix(axis, matrix, class_names, title):
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(class_names)), class_names, rotation=25)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    axis.set_title(title)


def main():
    x_train, x_test, y_train, y_test, _, class_names = load_iris_split()
    x_train, x_test, _, _ = standardize(x_train, x_test)
    k = 1

    custom_model = ParzenKNN(k=k).fit(x_train, y_train)
    reference_model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    reference_model.fit(x_train, y_train)

    custom_predictions = custom_model.predict(x_test)
    reference_predictions = reference_model.predict(x_test)
    custom_metrics = calculate_metrics(y_test, custom_predictions)
    reference_metrics = calculate_metrics(y_test, reference_predictions)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    draw_confusion_matrix(
        axes[0], custom_metrics["confusion_matrix"], class_names, "Собственный Parzen KNN"
    )
    draw_confusion_matrix(
        axes[1], reference_metrics["confusion_matrix"], class_names, "Sklearn KNN"
    )

    names = ["Accuracy", "Precision", "Recall", "F1"]
    keys = ["accuracy", "precision", "recall", "f1"]
    positions = np.arange(len(names))
    width = 0.36
    axes[2].bar(
        positions - width / 2,
        [custom_metrics[key] for key in keys],
        width,
        label="Собственный",
    )
    axes[2].bar(
        positions + width / 2,
        [reference_metrics[key] for key in keys],
        width,
        label="Sklearn",
    )
    axes[2].set_xticks(positions, names, rotation=20)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Значение")
    axes[2].set_title("Метрики на test")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "reference_comparison.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    for name, metrics in (
        ("Собственный Parzen KNN", custom_metrics),
        ("Sklearn KNN", reference_metrics),
    ):
        print(name)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro precision: {metrics['precision']:.4f}")
        print(f"  Macro recall: {metrics['recall']:.4f}")
        print(f"  Macro F1: {metrics['f1']:.4f}")
        print(f"  Матрица ошибок:\n{metrics['confusion_matrix']}")
    print(f"Совпавших предсказаний: {np.mean(custom_predictions == reference_predictions):.2%}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
