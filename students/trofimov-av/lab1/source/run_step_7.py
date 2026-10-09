from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_versicolor_virginica
from margins import decision_function
from optimizers import steepest_gradient_descent


def main() -> None:
    dataset = load_iris_versicolor_virginica()
    feature_mean = dataset.features.mean(axis=0)
    feature_scale = dataset.features.std(axis=0)
    features = (dataset.features - feature_mean) / feature_scale
    regularization = 0.1

    result = steepest_gradient_descent(
        features,
        dataset.labels,
        regularization=regularization,
        max_iterations=100,
        tolerance=1e-12,
    )

    design = np.column_stack((features, np.ones(len(features))))
    penalty = np.diag([regularization] * features.shape[1] + [0.0])
    exact_parameters = np.linalg.solve(
        design.T @ design / len(features) + penalty,
        design.T @ dataset.labels / len(features),
    )
    learned_parameters = np.append(result.weights, result.bias)
    parameter_difference = float(np.linalg.norm(learned_parameters - exact_parameters))

    scores = decision_function(features, result.weights, result.bias)
    predictions = np.where(scores >= 0.0, 1.0, -1.0)
    accuracy = float(np.mean(predictions == dataset.labels))
    original_weights = result.weights / feature_scale
    original_bias = result.bias - float(original_weights @ feature_mean)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        np.arange(len(result.loss_history)),
        result.loss_history,
        marker="o",
        markersize=3,
        label="Скорейший спуск",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Сходимость скорейшего спуска")
    axes[0].set_xlabel("Итерация")
    axes[0].set_ylabel("Функционал с учётом L2")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for label, name, color in zip(
        (-1.0, 1.0), dataset.class_names, ("tab:blue", "tab:orange"), strict=True
    ):
        mask = dataset.labels == label
        axes[1].scatter(
            dataset.features[mask, 0],
            dataset.features[mask, 1],
            color=color,
            label=name,
            edgecolor="black",
            alpha=0.8,
        )
    x_limits = (
        dataset.features[:, 0].min() - 0.2,
        dataset.features[:, 0].max() + 0.2,
    )
    y_limits = (
        dataset.features[:, 1].min() - 0.15,
        dataset.features[:, 1].max() + 0.15,
    )
    x_line = np.linspace(*x_limits)
    y_line = -(original_weights[0] * x_line + original_bias) / original_weights[1]
    axes[1].plot(x_line, y_line, "k--", label="Полученная граница")
    axes[1].set_xlim(x_limits)
    axes[1].set_ylim(y_limits)
    axes[1].set_title(f"Скорейший спуск, accuracy = {accuracy:.2%}")
    axes[1].set_xlabel(dataset.feature_names[0])
    axes[1].set_ylabel(dataset.feature_names[1])
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "steepest_descent.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Выполнено итераций: {len(result.step_history)}")
    print(f"Начальный функционал: {result.loss_history[0]:.12f}")
    print(f"Конечный функционал: {result.loss_history[-1]:.12f}")
    print(f"Точность классификации: {accuracy:.2%}")
    print(f"Отклонение от точного решения: {parameter_difference:.12e}")
    print(f"Минимальный шаг: {result.step_history.min():.6f}")
    print(f"Максимальный шаг: {result.step_history.max():.6f}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
