from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_split, standardize
from evaluation import calculate_metrics
from parzen_knn import ParzenKNN
from prototype_selection import select_prototypes


def measure_prediction_time(model, features, repeats=1000):
    model.predict(features)
    start = perf_counter()
    for _ in range(repeats):
        model.predict(features)
    return (perf_counter() - start) * 1000 / repeats


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
    selection = select_prototypes(x_train, y_train, k=1, noise_threshold=0.0)

    full_model = ParzenKNN(k=1).fit(x_train, y_train)
    reduced_model = ParzenKNN(k=1).fit(
        x_train[selection.prototype_indices], y_train[selection.prototype_indices]
    )

    full_predictions = full_model.predict(x_test)
    reduced_predictions = reduced_model.predict(x_test)
    full_metrics = calculate_metrics(y_test, full_predictions)
    reduced_metrics = calculate_metrics(y_test, reduced_predictions)
    full_time = measure_prediction_time(full_model, x_test)
    reduced_time = measure_prediction_time(reduced_model, x_test)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    draw_confusion_matrix(
        axes[0, 0], full_metrics["confusion_matrix"], class_names, "Все объекты"
    )
    draw_confusion_matrix(
        axes[0, 1], reduced_metrics["confusion_matrix"], class_names, "Только эталоны"
    )

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    positions = np.arange(len(metric_names))
    width = 0.36
    axes[1, 0].bar(
        positions - width / 2,
        [full_metrics[key] for key in metric_keys],
        width,
        label="Все объекты",
    )
    axes[1, 0].bar(
        positions + width / 2,
        [reduced_metrics[key] for key in metric_keys],
        width,
        label="Эталоны",
    )
    axes[1, 0].set_xticks(positions, metric_names, rotation=20)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Значение")
    axes[1, 0].set_title("Качество на test")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    sizes = [len(y_train), len(selection.prototype_indices)]
    axes[1, 1].bar(["Все объекты", "Эталоны"], sizes, color=["tab:blue", "tab:orange"])
    for index, value in enumerate(sizes):
        axes[1, 1].text(index, value + 2, str(value), ha="center")
    axes[1, 1].set_ylim(0, max(sizes) * 1.15)
    axes[1, 1].set_ylabel("Количество объектов")
    axes[1, 1].set_title("Размер обучающего набора")
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "prototype_comparison.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    for name, metrics, prediction_time in (
        ("Все объекты", full_metrics, full_time),
        ("Только эталоны", reduced_metrics, reduced_time),
    ):
        print(name)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro precision: {metrics['precision']:.4f}")
        print(f"  Macro recall: {metrics['recall']:.4f}")
        print(f"  Macro F1: {metrics['f1']:.4f}")
        print(f"  Время предсказания 30 объектов: {prediction_time:.4f} мс")
        print(f"  Матрица ошибок:\n{metrics['confusion_matrix']}")
    print(f"Ускорение предсказания: {full_time / reduced_time:.2f} раза")
    print(f"Совпавших предсказаний: {np.mean(full_predictions == reduced_predictions):.2%}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
