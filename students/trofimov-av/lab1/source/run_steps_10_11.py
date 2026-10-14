from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

from data import load_iris_versicolor_virginica
from evaluation import (
    ClassificationMetrics,
    classification_metrics,
    roc_curve_points,
)
from margins import decision_function
from optimizers import OptimizationResult, stochastic_gradient_descent_momentum
from training import correlation_initialization, multistart_momentum


def evaluate_custom(
    features: np.ndarray,
    labels: np.ndarray,
    result: OptimizationResult,
) -> tuple[ClassificationMetrics, np.ndarray]:
    scores = decision_function(features, result.weights, result.bias)
    predictions = np.where(scores >= 0.0, 1.0, -1.0)
    return classification_metrics(labels, predictions, scores), scores


def print_metrics(name: str, metrics: ClassificationMetrics) -> None:
    interval = metrics.accuracy_interval
    print(name)
    print(f"  Accuracy: {metrics.accuracy:.2%} [{interval[0]:.2%}; {interval[1]:.2%}]")
    print(f"  Precision: {metrics.precision:.6f}")
    print(f"  Recall: {metrics.recall:.6f}")
    print(f"  Specificity: {metrics.specificity:.6f}")
    print(f"  F1: {metrics.f1:.6f}")
    print(f"  ROC-AUC: {metrics.roc_auc:.6f}")
    print(f"  Квадратичная потеря: {metrics.quadratic_loss:.6f}")
    print(
        f"  Матрица ошибок: TN={metrics.true_negative}, FP={metrics.false_positive}, "
        f"FN={metrics.false_negative}, TP={metrics.true_positive}"
    )


def draw_confusion_matrix(
    axis: plt.Axes,
    metrics: ClassificationMetrics,
    title: str,
) -> None:
    matrix = np.array(
        [
            [metrics.true_negative, metrics.false_positive],
            [metrics.false_negative, metrics.true_positive],
        ]
    )
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks((0, 1), ("versicolor", "virginica"))
    axis.set_yticks((0, 1), ("versicolor", "virginica"))
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    axis.set_title(title)


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
    test_features = dataset.features[test_indices]
    train_labels = dataset.labels[train_indices]
    test_labels = dataset.labels[test_indices]
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    scaled_train = (train_features - feature_mean) / feature_scale
    scaled_test = (test_features - feature_mean) / feature_scale

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
    margin_result = stochastic_gradient_descent_momentum(
        scaled_train,
        train_labels,
        initial_weights=multistart.best_initial_weights,
        initial_bias=multistart.best_initial_bias,
        learning_rate=0.01,
        momentum=0.9,
        epochs=100,
        smoothing=0.02,
        regularization=0.1,
        presentation="margin",
        random_state=multistart.best_random_state,
    )

    custom_results = (
        ("Корреляционная", correlation_result),
        ("Мультистарт", multistart.best_result),
        ("Мультистарт + |M|", margin_result),
    )
    evaluated_custom: dict[str, tuple[ClassificationMetrics, np.ndarray]] = {}
    for name, result in custom_results:
        evaluated_custom[name] = evaluate_custom(scaled_test, test_labels, result)
        print_metrics(name, evaluated_custom[name][0])

    ridge_alpha = len(train_labels) * 0.1
    reference = RidgeClassifier(alpha=ridge_alpha, fit_intercept=True)
    reference.fit(scaled_train, train_labels)
    reference_scores = np.asarray(
        reference.decision_function(scaled_test), dtype=np.float64
    )
    reference_predictions = np.asarray(reference.predict(scaled_test), dtype=np.float64)
    reference_metrics = classification_metrics(
        test_labels, reference_predictions, reference_scores
    )
    print_metrics("Эталонный RidgeClassifier", reference_metrics)
    print(f"Эквивалентный sklearn alpha: {ridge_alpha:.6f}")

    best_metrics, best_scores = evaluated_custom["Мультистарт"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    draw_confusion_matrix(axes[0, 0], best_metrics, "Собственная модель")
    draw_confusion_matrix(axes[0, 1], reference_metrics, "RidgeClassifier")

    custom_fpr, custom_tpr = roc_curve_points(test_labels, best_scores)
    reference_fpr, reference_tpr = roc_curve_points(test_labels, reference_scores)
    axes[1, 0].plot(
        custom_fpr,
        custom_tpr,
        marker="o",
        label=f"Собственная, AUC={best_metrics.roc_auc:.3f}",
    )
    axes[1, 0].plot(
        reference_fpr,
        reference_tpr,
        marker="s",
        label=f"RidgeClassifier, AUC={reference_metrics.roc_auc:.3f}",
    )
    axes[1, 0].plot((0, 1), (0, 1), "k--", label="Случайный классификатор")
    axes[1, 0].set_title("ROC-кривые на тестовой выборке")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    metric_names = ("Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC")
    custom_values = (
        best_metrics.accuracy,
        best_metrics.precision,
        best_metrics.recall,
        best_metrics.specificity,
        best_metrics.f1,
        best_metrics.roc_auc,
    )
    reference_values = (
        reference_metrics.accuracy,
        reference_metrics.precision,
        reference_metrics.recall,
        reference_metrics.specificity,
        reference_metrics.f1,
        reference_metrics.roc_auc,
    )
    positions = np.arange(len(metric_names))
    width = 0.38
    axes[1, 1].bar(
        positions - width / 2,
        custom_values,
        width,
        label="Собственная",
    )
    axes[1, 1].bar(
        positions + width / 2,
        reference_values,
        width,
        label="RidgeClassifier",
    )
    axes[1, 1].set_xticks(positions, metric_names, rotation=30, ha="right")
    axes[1, 1].set_ylim(0.0, 1.08)
    axes[1, 1].set_title("Сравнение метрик")
    axes[1, 1].set_ylabel("Значение")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).resolve().parents[1] / "figures" / "evaluation.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"График сохранён: {output_path}")


if __name__ == "__main__":
    main()
