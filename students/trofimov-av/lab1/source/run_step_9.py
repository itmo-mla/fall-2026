from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from data import load_iris_versicolor_virginica
from margins import decision_function, functional_margin
from optimizers import OptimizationResult, stochastic_gradient_descent_momentum
from training import correlation_initialization, multistart_momentum


def accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    result: OptimizationResult,
) -> float:
    scores = decision_function(features, result.weights, result.bias)
    return float(np.mean(np.where(scores >= 0.0, 1.0, -1.0) == labels))


def boundary_in_original_scale(
    result: OptimizationResult,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    weights = result.weights / feature_scale
    bias = result.bias - float(weights @ feature_mean)
    return weights, bias


def main() -> None:
    dataset = load_iris_versicolor_virginica()
    indices = np.arange(len(dataset.labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=dataset.labels,
    )
    train_features = dataset.features[train_indices]
    train_labels = dataset.labels[train_indices]
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    scaled_train = (train_features - feature_mean) / feature_scale

    optimizer_parameters = dict(
        learning_rate=0.01,
        momentum=0.9,
        epochs=100,
        smoothing=0.02,
        regularization=0.1,
        presentation="random",
    )

    correlation_weights, correlation_bias = correlation_initialization(
        scaled_train, train_labels
    )
    correlation_result = stochastic_gradient_descent_momentum(
        scaled_train,
        train_labels,
        initial_weights=correlation_weights,
        initial_bias=correlation_bias,
        random_state=42,
        **optimizer_parameters,
    )

    multistart = multistart_momentum(
        scaled_train,
        train_labels,
        starts=20,
        initialization_scale=0.5,
        random_state=42,
        **optimizer_parameters,
    )

    margin_parameters = optimizer_parameters | {"presentation": "margin"}
    margin_result = stochastic_gradient_descent_momentum(
        scaled_train,
        train_labels,
        initial_weights=multistart.best_initial_weights,
        initial_bias=multistart.best_initial_bias,
        random_state=multistart.best_random_state,
        **margin_parameters,
    )

    results = (
        ("Корреляционная", correlation_result, "tab:green"),
        ("Мультистарт", multistart.best_result, "tab:purple"),
        ("Мультистарт + |M|", margin_result, "black"),
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, result, color in results:
        axes[0].plot(
            np.arange(1, len(result.epoch_loss) + 1),
            result.epoch_loss,
            label=name,
            color=color,
        )
    axes[0].set_title("Обучающий функционал")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Функционал с учётом L2")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        np.arange(1, len(multistart.final_losses) + 1),
        multistart.final_losses,
        color=np.where(
            np.arange(len(multistart.final_losses)) == multistart.best_start,
            "tab:green",
            "tab:blue",
        ),
    )
    axes[1].set_title("Результаты случайных стартов")
    axes[1].set_xlabel("Номер старта")
    axes[1].set_ylabel("Конечный функционал")
    axes[1].annotate(
        "Лучший",
        xy=(multistart.best_start + 1, multistart.final_losses[multistart.best_start]),
        xytext=(multistart.best_start + 1, multistart.final_losses.max() + 0.005),
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "tab:green"},
    )
    axes[1].grid(axis="y", alpha=0.25)

    for label, name, color in zip(
        (-1.0, 1.0), dataset.class_names, ("tab:blue", "tab:orange"), strict=True
    ):
        mask = train_labels == label
        axes[2].scatter(
            train_features[mask, 0],
            train_features[mask, 1],
            color=color,
            label=f"{name}, train",
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
    for name, result, color in results:
        weights, bias = boundary_in_original_scale(
            result, feature_mean, feature_scale
        )
        y_line = -(weights[0] * x_line + bias) / weights[1]
        axes[2].plot(x_line, y_line, linestyle="--", color=color, label=name)
    axes[2].set_xlim(x_limits)
    axes[2].set_ylim(y_limits)
    axes[2].set_title("Границы на обучающей выборке")
    axes[2].set_xlabel(dataset.feature_names[0])
    axes[2].set_ylabel(dataset.feature_names[1])
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "training_modes.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Обучающих объектов: {len(train_indices)}")
    print(f"Отложенных объектов: {len(test_indices)}")
    print(f"Лучший случайный старт: {multistart.best_start + 1} из 20")
    print(
        "Диапазон функционала мультистарта: "
        f"{multistart.final_losses.min():.6f}–{multistart.final_losses.max():.6f}"
    )
    for name, result, _ in results:
        margins = functional_margin(
            scaled_train, train_labels, result.weights, result.bias
        )
        print(name)
        print(f"  Функционал: {result.epoch_loss[-1]:.6f}")
        print(f"  Train accuracy: {accuracy(scaled_train, train_labels, result):.2%}")
        print(f"  Неположительных отступов: {np.count_nonzero(margins <= 0.0)}")
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
