from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from data import prepare_validation_split
from model import SVM


@dataclass
class ModelComparison:
    kernel: str
    own: SVM
    reference: SVC


def evaluate_parameter_grid(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> pd.DataFrame:
    validation_rows = []
    for kernel in ("linear", "rbf"):
        gamma_values = [0.1] if kernel == "linear" else [0.01, 0.1]
        for C in (0.1, 1.0, 10.0):
            for gamma in gamma_values:
                classifier = SVM(C=C, kernel=kernel, gamma=gamma)
                classifier.fit(fit_features, fit_labels)
                predictions = classifier.predict(validation_features)
                validation_rows.append({
                    "kernel": kernel,
                    "C": C,
                    "gamma": gamma,
                    "f1": f1_score(validation_labels, predictions),
                    "errors": np.count_nonzero(predictions != validation_labels),
                })
    return pd.DataFrame(validation_rows)


def best_parameters_by_kernel(validation_results: pd.DataFrame) -> dict[str, dict]:
    best_parameters = {}
    for kernel in ("linear", "rbf"):
        candidates = validation_results[validation_results["kernel"] == kernel]
        ordered = candidates.sort_values(
            ["f1", "C", "gamma"], ascending=[False, True, True]
        )
        best = ordered.iloc[0]
        best_parameters[kernel] = {"C": float(best.C), "gamma": float(best.gamma)}
    return best_parameters


def select_parameters(
    training_features: np.ndarray,
    training_labels: np.ndarray,
) -> tuple[dict[str, dict], pd.DataFrame]:
    fit_features, validation_features, fit_labels, validation_labels = (
        prepare_validation_split(training_features, training_labels)
    )
    validation_results = evaluate_parameter_grid(
        fit_features, fit_labels, validation_features, validation_labels
    )
    best_parameters = best_parameters_by_kernel(validation_results)
    return best_parameters, validation_results


def train_models(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    parameters: dict[str, dict],
) -> list[ModelComparison]:
    results = []
    for kernel, settings in parameters.items():
        own = SVM(kernel=kernel, **settings).fit(training_features, training_labels)
        reference = SVC(kernel=kernel, **settings, tol=1e-8, shrinking=False)
        reference.fit(training_features, training_labels)
        results.append(ModelComparison(kernel=kernel, own=own, reference=reference))
    return results


def train_boundary_models(
    training_features: np.ndarray,
    training_labels: np.ndarray,
) -> dict[str, SVM]:
    boundary_features = training_features[:, :2]
    return {
        kernel: SVM(C=1, kernel=kernel, gamma=0.1).fit(
            boundary_features, training_labels
        )
        for kernel in ("linear", "rbf")
    }
