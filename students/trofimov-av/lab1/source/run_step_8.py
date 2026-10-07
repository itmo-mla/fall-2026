from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_iris_versicolor_virginica
from margins import centroid_linear_rule, decision_function, functional_margin
from optimizers import OptimizationResult, stochastic_gradient_descent_momentum


def train(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    presentation: str,
) -> OptimizationResult:
    return stochastic_gradient_descent_momentum(
        features,
        labels,
        initial_weights=weights,
        initial_bias=bias,
        learning_rate=0.01,
        momentum=0.9,
        epochs=100,
        smoothing=0.02,
        regularization=0.1,
        presentation=presentation,
        random_state=42,
    )


def main() -> None:
    dataset = load_iris_versicolor_virginica()
    feature_mean = dataset.features.mean(axis=0)
    feature_scale = dataset.features.std(axis=0)
    features = (dataset.features - feature_mean) / feature_scale
    initial_weights, initial_bias = centroid_linear_rule(features, dataset.labels)

    random_result = train(
        features,
        dataset.labels,
        initial_weights,
        initial_bias,
        "random",
    )
    margin_result = train(
        features,
        dataset.labels,
        initial_weights,
        initial_bias,
        "margin",
    )
    results = (
        ("Случайный порядок", random_result),
        ("По возрастанию |M|", margin_result),
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, result in results:
        axes[0].plot(
            np.arange(1, len(result.epoch_loss) + 1),
            result.epoch_loss,
            label=name,
        )
    axes[0].set_title("Сравнение порядка предъявления")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Функционал с учётом L2")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    final_margins = functional_margin(
        features,
        dataset.labels,
        margin_result.weights,
        margin_result.bias,
    )
    order = np.argsort(np.abs(final_margins))
    ordered_margins = final_margins[order]
    positions = np.arange(len(final_margins))
    positive = ordered_margins > 0.0
    axes[1].bar(
        positions[positive],
        ordered_margins[positive],
        color="tab:blue",
        label="M > 0",
    )
    axes[1].bar(
        positions[~positive],
        ordered_margins[~positive],
        color="tab:red",
        label="M ≤ 0",
    )
    axes[1].axhline(0.0, color="black", linestyle="--", label="M = 0")
    axes[1].set_title("Отступы в порядке возрастания |M|")
    axes[1].set_xlabel("Позиция объекта в порядке предъявления")
    axes[1].set_ylabel("Функциональный отступ")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "margin_order.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    for name, result in results:
        scores = decision_function(features, result.weights, result.bias)
        predictions = np.where(scores >= 0.0, 1.0, -1.0)
        accuracy = float(np.mean(predictions == dataset.labels))
        margins = functional_margin(
            features, dataset.labels, result.weights, result.bias
        )
        print(name)
        print(f"  Конечный функционал: {result.epoch_loss[-1]:.6f}")
        print(f"  Точность: {accuracy:.2%}")
        print(f"  Неположительных отступов: {np.count_nonzero(margins <= 0.0)}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
