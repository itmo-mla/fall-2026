from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_versicolor_virginica
from losses import l2_penalty, squared_loss
from margins import decision_function
from optimizers import OptimizationResult, stochastic_gradient_descent_momentum


def train(
    features: np.ndarray,
    labels: np.ndarray,
    regularization: float,
) -> OptimizationResult:
    return stochastic_gradient_descent_momentum(
        features,
        labels,
        learning_rate=0.01,
        momentum=0.9,
        epochs=100,
        smoothing=0.02,
        regularization=regularization,
        shuffle=True,
        random_state=42,
    )


def main() -> None:
    dataset = load_iris_versicolor_virginica()
    feature_mean = dataset.features.mean(axis=0)
    feature_scale = dataset.features.std(axis=0)
    features = (dataset.features - feature_mean) / feature_scale
    regularization = 0.1

    plain_result = train(features, dataset.labels, 0.0)
    regularized_result = train(features, dataset.labels, regularization)
    results = (("Без L2", plain_result), (f"L2, λ={regularization}", regularized_result))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, result in results:
        axes[0].plot(
            np.arange(1, len(result.epoch_loss) + 1),
            result.epoch_loss,
            label=name,
        )
    axes[0].set_title("Функционал качества по эпохам")
    axes[0].set_xlabel("Эпоха")
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
            alpha=0.75,
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
    for name, result in results:
        original_weights = result.weights / feature_scale
        original_bias = result.bias - float(original_weights @ feature_mean)
        y_line = -(original_weights[0] * x_line + original_bias) / original_weights[1]
        axes[1].plot(x_line, y_line, linestyle="--", label=name)
    axes[1].set_xlim(x_limits)
    axes[1].set_ylim(y_limits)
    axes[1].set_title("Влияние L2 на разделяющую границу")
    axes[1].set_xlabel(dataset.feature_names[0])
    axes[1].set_ylabel(dataset.feature_names[1])
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "l2_regularization.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    for name, result in results:
        scores = decision_function(features, result.weights, result.bias)
        predictions = np.where(scores >= 0.0, 1.0, -1.0)
        accuracy = float(np.mean(predictions == dataset.labels))
        data_loss = squared_loss(features, dataset.labels, result.weights, result.bias)
        penalty = l2_penalty(result.weights, regularization if name != "Без L2" else 0.0)
        print(name)
        print(f"  Потеря на данных: {data_loss:.6f}")
        print(f"  L2-штраф: {penalty:.6f}")
        print(f"  Норма весов: {np.linalg.norm(result.weights):.6f}")
        print(f"  Точность: {accuracy:.2%}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
