from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_versicolor_virginica
from losses import squared_loss
from margins import decision_function
from optimizers import stochastic_gradient_descent_momentum


def main() -> None:
    dataset = load_iris_versicolor_virginica()
    feature_mean = dataset.features.mean(axis=0)
    feature_scale = dataset.features.std(axis=0)
    scaled_features = (dataset.features - feature_mean) / feature_scale
    initial_loss = squared_loss(
        scaled_features,
        dataset.labels,
        np.zeros(scaled_features.shape[1]),
        0.0,
    )

    result = stochastic_gradient_descent_momentum(
        scaled_features,
        dataset.labels,
        learning_rate=0.01,
        momentum=0.9,
        epochs=100,
        smoothing=0.02,
        shuffle=True,
        random_state=42,
    )

    predictions = np.where(
        decision_function(scaled_features, result.weights, result.bias) >= 0.0,
        1.0,
        -1.0,
    )
    accuracy = float(np.mean(predictions == dataset.labels))
    original_weights = result.weights / feature_scale
    original_bias = result.bias - float(original_weights @ feature_mean)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    iterations = np.arange(1, len(result.recursive_quality) + 1)
    axes[0].plot(iterations, result.recursive_quality, label="Рекуррентная оценка")
    axes[0].plot(
        np.arange(1, len(result.epoch_loss) + 1) * len(dataset.labels),
        result.epoch_loss,
        color="tab:red",
        label="Эмпирический риск после эпохи",
    )
    axes[0].set_title("Сходимость SGD с инерцией")
    axes[0].set_xlabel("Итерация")
    axes[0].set_ylabel("Квадратичная потеря")
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
    axes[1].plot(x_line, y_line, "k--", label="Граница после SGD")
    axes[1].set_xlim(x_limits)
    axes[1].set_ylim(y_limits)
    axes[1].set_title(f"Результат обучения, accuracy = {accuracy:.2%}")
    axes[1].set_xlabel(dataset.feature_names[0])
    axes[1].set_ylabel(dataset.feature_names[1])
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "sgd_momentum.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Начальный эмпирический риск: {initial_loss:.6f}")
    print(f"Конечный эмпирический риск: {result.epoch_loss[-1]:.6f}")
    print(f"Точность классификации: {accuracy:.2%}")
    print(f"Веса в исходном масштабе: {original_weights}")
    print(f"Смещение в исходном масштабе: {original_bias:.6f}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
