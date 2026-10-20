import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from experiments import ModelComparison
from model import SVM, kernel_matrix


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predictions = np.where(scores >= 0, 1, -1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "errors": np.count_nonzero(labels != predictions),
        "false_negatives": np.count_nonzero((labels == 1) & (predictions == -1)),
        "false_positives": np.count_nonzero((labels == -1) & (predictions == 1)),
    }


def primal_objective(
    classifier: SVM,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    gram_matrix = kernel_matrix(features, features, classifier.kernel, classifier.gamma)
    signed_multipliers = classifier.multipliers * labels
    scores_without_intercept = np.dot(gram_matrix, signed_multipliers)
    squared_norm = np.dot(signed_multipliers, scores_without_intercept)
    margins = labels * (scores_without_intercept + classifier.intercept)
    hinge_losses = np.maximum(0, 1 - margins)
    return float(0.5 * squared_norm + classifier.C * hinge_losses.sum())


def compare_solution(
    result: ModelComparison,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    own_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> dict:
    classifier = result.own
    primal_value = primal_objective(classifier, training_features, training_labels)
    prediction_differences = np.count_nonzero((own_scores >= 0) != (reference_scores >= 0))
    return {
        "kernel": result.kernel,
        "equality_error": abs(np.dot(classifier.multipliers, training_labels)),
        "duality_gap": primal_value - classifier.dual_value,
        "max_score_difference": np.max(np.abs(own_scores - reference_scores)),
        "prediction_differences": prediction_differences,
    }


def evaluate_results(
    results: list[ModelComparison],
    training_features: np.ndarray,
    training_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    diagnostic_rows = []
    for result in results:
        own_scores = result.own.decision_function(test_features)
        reference_scores = result.reference.decision_function(test_features)
        models = [
            ("Собственная", own_scores, len(result.own.support_indices)),
            ("sklearn SVC", reference_scores, len(result.reference.support_)),
        ]
        for name, scores, support_count in models:
            metric_rows.append({
                "kernel": result.kernel,
                "model": name,
                "support_count": support_count,
                **classification_metrics(test_labels, scores),
            })
        diagnostic_rows.append(compare_solution(
            result, training_features, training_labels, own_scores, reference_scores
        ))
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)
