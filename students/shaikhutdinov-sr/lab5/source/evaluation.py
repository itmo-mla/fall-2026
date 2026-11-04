from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from data import Dataset, PreparedData
from model import log_loss, loss_gradient, predict_probabilities, residuals_and_variances
from optimization import FitResult


@dataclass
class EvaluationResults:
    fits: dict[str, FitResult]
    probabilities: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    equivalence: pd.DataFrame
    metrics: pd.DataFrame
    confusion: np.ndarray
    precision_recall: pd.DataFrame
    irls_weights: pd.DataFrame
    calibration: pd.DataFrame
    trajectory_difference: float


def predict_models(
    test_data: Dataset,
    fits: dict[str, FitResult],
    reference: LogisticRegression,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    features = test_data.features
    probabilities = {
        name: predict_probabilities(features, fit.weights)
        for name, fit in fits.items()
    }
    predictions = {
        name: (values > 0.5).astype(int)
        for name, values in probabilities.items()
    }
    probabilities["sklearn"] = reference.predict_proba(test_data.features[:, 1:])[:, 1]
    predictions["sklearn"] = reference.predict(test_data.features[:, 1:])
    return probabilities, predictions


def compare_with_reference(
    training_data: Dataset,
    weights: dict[str, np.ndarray],
    iterations: dict[str, int],
    probabilities: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    features = training_data.features
    labels = training_data.labels
    reference_weights = weights["sklearn"]
    reference_probabilities = probabilities["sklearn"]
    reference_predictions = predictions["sklearn"]
    reference_loss = log_loss(np.dot(features, reference_weights), labels)

    rows = []
    for name, model_weights in weights.items():
        scores = np.dot(features, model_weights)
        residuals, _ = residuals_and_variances(scores, labels)
        gradient = loss_gradient(features, residuals)
        train_loss = log_loss(scores, labels)
        rows.append({
            "method": name,
            "iterations": iterations[name],
            "train_loss": train_loss,
            "gradient_norm": np.linalg.norm(gradient, ord=np.inf),
            "max_weight_difference": np.max(np.abs(model_weights - reference_weights)),
            "max_probability_difference": np.max(np.abs(
                probabilities[name] - reference_probabilities
            )),
            "loss_difference": abs(train_loss - reference_loss),
            "prediction_disagreements": np.count_nonzero(
                predictions[name] != reference_predictions
            ),
        })
    return pd.DataFrame(rows).set_index("method")


def evaluate_test_metrics(
    test_data: Dataset,
    weights: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    features = test_data.features
    labels = test_data.labels
    rows = []
    for name, model_weights in weights.items():
        rows.append({
            "method": name,
            "precision": precision_score(labels, predictions[name]),
            "recall": recall_score(labels, predictions[name]),
            "f1": f1_score(labels, predictions[name]),
            "average_precision": average_precision_score(labels, probabilities[name]),
            "log_loss": log_loss(np.dot(features, model_weights), labels),
            "errors": np.count_nonzero(labels != predictions[name]),
        })
    return pd.DataFrame(rows).set_index("method")


def compare_weight_histories(first: FitResult, second: FitResult) -> float:
    common_iterations = min(len(first.weight_history), len(second.weight_history))
    weight_differences = (
        first.weight_history[:common_iterations]
        - second.weight_history[:common_iterations]
    )
    return float(np.max(np.abs(weight_differences)))


def summarize_irls_weights(training_data: Dataset, fit: FitResult) -> pd.DataFrame:
    rows = []
    for iteration, weights in enumerate(fit.weight_history):
        scores = np.dot(training_data.features, weights)
        _, object_weights = residuals_and_variances(scores, training_data.labels)
        percentile_10, median, percentile_90, percentile_99 = np.quantile(
            object_weights, [0.1, 0.5, 0.9, 0.99],
        )
        rows.append({
            "iteration": iteration,
            "percentile_10": percentile_10,
            "median": median,
            "percentile_90": percentile_90,
            "percentile_99": percentile_99,
        })
    return pd.DataFrame(rows).set_index("iteration")


def evaluate_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    bin_edges = np.arange(n_bins + 1) / n_bins
    bin_indices = np.digitize(probabilities, bin_edges[1:-1])
    z = norm.ppf(0.975)
    rows = []
    for bin_index in range(n_bins):
        selected = bin_indices == bin_index
        count = int(selected.sum())
        if count == 0:
            continue
        positives = int(labels[selected].sum())
        observed_fraction = positives / count
        # Интервал Уилсона для доли положительного класса, включая доли 0 и 1.
        denominator = 1 + z**2 / count
        center = (observed_fraction + z**2 / (2 * count)) / denominator
        radius = z * np.sqrt(observed_fraction * (1 - observed_fraction) / count + z**2 / (4 * count**2)) / denominator
        rows.append({
            "bin_lower": bin_edges[bin_index],
            "bin_upper": bin_edges[bin_index + 1],
            "count": count,
            "positives": positives,
            "mean_probability": probabilities[selected].mean(),
            "observed_fraction": observed_fraction,
            "interval_lower": max(0.0, center - radius),
            "interval_upper": min(1.0, center + radius),
        })
    return pd.DataFrame(rows)


def evaluate_results(
    data: PreparedData,
    fits: dict[str, FitResult],
    reference: LogisticRegression,
) -> EvaluationResults:
    weights = {name: fit.weights for name, fit in fits.items()}
    weights["sklearn"] = np.concatenate([reference.intercept_, reference.coef_[0]])
    iterations = {name: fit.iterations for name, fit in fits.items()}
    iterations["sklearn"] = int(reference.n_iter_[0])

    probabilities, predictions = predict_models(data.test, fits, reference)
    equivalence = compare_with_reference(
        data.train, weights, iterations, probabilities, predictions,
    )
    metrics = evaluate_test_metrics(data.test, weights, probabilities, predictions)
    trajectory_difference = compare_weight_histories(fits["Ньютон–Рафсон"], fits["IRLS"])
    confusion = confusion_matrix(
        data.test.labels, predictions["Ньютон–Рафсон"], labels=[0, 1],
    )
    precision, recall, _ = precision_recall_curve(
        data.test.labels, probabilities["Ньютон–Рафсон"],
    )

    return EvaluationResults(
        fits=fits,
        probabilities=probabilities,
        predictions=predictions,
        equivalence=equivalence,
        metrics=metrics,
        confusion=confusion,
        precision_recall=pd.DataFrame({"precision": precision, "recall": recall}),
        irls_weights=summarize_irls_weights(data.train, fits["IRLS"]),
        calibration=evaluate_calibration(data.test.labels, probabilities["Ньютон–Рафсон"]),
        trajectory_difference=trajectory_difference,
    )
