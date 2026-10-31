from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from model import ParzenKNN, loo_risks
from selection import select_prototypes


@dataclass
class ExperimentResult:
    name: str
    classifier: ParzenKNN | KNeighborsClassifier
    loo_curve: pd.Series
    neighbor_count: int


@dataclass
class ExperimentResults:
    full: ExperimentResult
    reference: ExperimentResult
    reduced: ExperimentResult
    prototype_indices: np.ndarray
    selection_history: pd.DataFrame


def run_parzen_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
) -> ExperimentResult:
    risks = loo_risks(x_train, y_train, k_values)
    loo_curve = pd.Series(risks, index=k_values, name="risk").sort_index()
    k = int(loo_curve.idxmin())
    classifier = ParzenKNN(x_neighbors=x_train, y_neighbors=y_train, k=k)
    return ExperimentResult(
        name="Парзен",
        classifier=classifier,
        loo_curve=loo_curve,
        neighbor_count=len(y_train),
    )


def run_reference_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
) -> ExperimentResult:
    classifier = KNeighborsClassifier()
    classifier.fit(x_train, y_train)
    risks = []
    for k in k_values:
        classifier.n_neighbors = k
        loo_predictions = classifier.predict(None)
        risk = np.mean(loo_predictions != y_train)
        risks.append(risk)

    loo_curve = pd.Series(risks, index=k_values, name="risk").sort_index()
    k = int(loo_curve.idxmin())
    classifier.n_neighbors = k
    return ExperimentResult(
        name="sklearn KNN",
        classifier=classifier,
        loo_curve=loo_curve,
        neighbor_count=len(y_train),
    )


def run_prototype_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
) -> tuple[ExperimentResult, np.ndarray, pd.DataFrame]:
    prototype_indices, selection_history = select_prototypes(x_train, y_train)
    x_prototypes = x_train[prototype_indices]
    y_prototypes = y_train[prototype_indices]

    max_k = len(prototype_indices) - 2
    reduced_k_values = [k for k in k_values if k <= max_k]
    if not reduced_k_values:
        raise ValueError("Для LOO окна Парзена нужны хотя бы три эталона")

    risks = loo_risks(x_train, y_train, reduced_k_values, prototype_indices)
    loo_curve = pd.Series(risks, index=reduced_k_values, name="risk").sort_index()
    k = int(loo_curve.idxmin())
    classifier = ParzenKNN(
        x_neighbors=x_prototypes,
        y_neighbors=y_prototypes,
        k=k,
    )
    result = ExperimentResult(
        name="Парзен с эталонами",
        classifier=classifier,
        loo_curve=loo_curve,
        neighbor_count=len(prototype_indices),
    )
    selection_history["parzen_risk"] = [
        loo_risks(x_train, y_train, [1], indices)[0]
        for indices in selection_history["prototype_indices"]
    ]
    return result, prototype_indices, selection_history


def run_experiments(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
) -> ExperimentResults:
    full = run_parzen_experiment(x_train, y_train, k_values)
    reference = run_reference_experiment(x_train, y_train, k_values)
    reduced, prototype_indices, selection_history = run_prototype_experiment(
        x_train, y_train, k_values
    )
    return ExperimentResults(
        full=full,
        reference=reference,
        reduced=reduced,
        prototype_indices=prototype_indices,
        selection_history=selection_history,
    )
