from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import Dataset
from experiments import (
    ExperimentResult,
    correlation_initialization,
    random_initialization,
)
from model import (
    calculate_margins,
    decision_function,
    l2_penalty,
    mean_squared_loss,
    predict_labels,
)
from optimization import FitResult


@dataclass
class EvaluationResults:
    best_run: str
    validation_runs: pd.DataFrame
    validation_summary: pd.DataFrame
    learning_histories: dict[str, pd.DataFrame]
    test_scores: dict[str, np.ndarray]
    test_metrics: pd.DataFrame
    initial_margins: dict[str, np.ndarray]
    final_margins: dict[str, np.ndarray]
    margin_summary: pd.DataFrame


def evaluate_scores(
    labels: np.ndarray,
    decision_values: np.ndarray,
) -> dict[str, float]:
    predictions = predict_labels(decision_values)
    red_labels = labels == 1
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision_red": precision_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
        "recall_red": recall_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
        "f1_red": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(red_labels, decision_values),
        "average_precision": average_precision_score(red_labels, decision_values),
    }


def evaluate_models(
    labels: np.ndarray,
    model_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for name, scores in model_scores.items():
        row = {"model": name}
        metrics = evaluate_scores(labels, scores)
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_experiments(
    runs: dict[str, ExperimentResult],
    validation_data: Dataset,
) -> pd.DataFrame:
    rows = []
    for name, run in runs.items():
        weights = run.result.weights
        scores = decision_function(validation_data.features, weights)
        row = {
            "run": name,
            "initialization": run.initialization,
            "sampling": run.sampling,
            "seed": run.seed,
            "epochs": len(run.result.epoch_weights),
            "stop_reason": run.result.stop_reason,
        }
        row.update(evaluate_scores(validation_data.labels, scores))
        errors = scores - validation_data.labels
        row["validation_loss"] = float((errors ** 2).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_learning_history(
    training_data: Dataset,
    validation_data: Dataset,
    result: FitResult,
    regularization: float,
) -> pd.DataFrame:
    rows = []
    weight_changes = np.linalg.norm(np.diff(result.epoch_weights, axis=0), axis=1)
    weight_scales = np.maximum(1.0, np.linalg.norm(result.epoch_weights[:-1], axis=1))
    relative_changes = np.concatenate(([np.nan], weight_changes / weight_scales))
    for epoch, weights in enumerate(result.epoch_weights):
        train_loss = mean_squared_loss(
            training_data.features, training_data.labels, weights
        )
        validation_scores = decision_function(validation_data.features, weights)
        rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": float(
                ((validation_scores - validation_data.labels) ** 2).mean()
            ),
            "validation_f1": f1_score(
                validation_data.labels,
                predict_labels(validation_scores),
                pos_label=1,
                zero_division=0,
            ),
            "train_objective": train_loss + l2_penalty(weights, regularization),
            "Q": result.quality_history[epoch],
            "relative_weight_change": relative_changes[epoch],
        })
    return pd.DataFrame(rows)


def select_best_run(validation_runs: pd.DataFrame) -> pd.Series:
    ordered = validation_runs.sort_values(
        ["f1_red", "validation_loss"], ascending=[False, True]
    )
    return ordered.iloc[0]


def summarize_experiments(validation_runs: pd.DataFrame) -> pd.DataFrame:
    is_steepest = validation_runs["run"] == "steepest_uniform"
    standard_runs = validation_runs[~is_steepest]
    steepest_run = validation_runs[is_steepest]

    winners = []
    groups = standard_runs.groupby(["initialization", "sampling"], sort=False)
    for _, group in groups:
        winners.append(select_best_run(group))

    best_standard_runs = pd.DataFrame(winners)
    return pd.concat([best_standard_runs, steepest_run], ignore_index=True)


def summarize_margins(margins_by_split: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, margins in margins_by_split.items():
        rows.append({
            "split": name,
            "mean_margin": float(margins.mean()),
            "negative_fraction": float((margins < 0).mean()),
            "near_boundary_fraction": float((np.abs(margins) < 0.1).mean()),
        })
    return pd.DataFrame(rows)


def evaluate_results(
    datasets: dict[str, Dataset],
    runs: dict[str, ExperimentResult],
    reference: RidgeClassifier,
    regularization: float,
) -> EvaluationResults:
    train_data = datasets["train"]
    validation_data = datasets["validation"]
    test_data = datasets["test"]

    validation_runs = evaluate_experiments(runs, validation_data)
    validation_summary = summarize_experiments(validation_runs)
    best_name = select_best_run(validation_runs)["run"]
    best_weights = runs[best_name].result.weights

    learning_histories = {
        name: evaluate_learning_history(
            train_data, validation_data, runs[name].result, regularization
        )
        for name in validation_summary["run"]
    }

    test_scores = {
        "Собственная реализация": decision_function(test_data.features, best_weights),
        "RidgeClassifier": reference.decision_function(test_data.features[:, 1:]),
        "Всегда белое": np.full(len(test_data.labels), -1.0),
    }
    initial_weights = {
        "Корреляционная": correlation_initialization(
            train_data.features, train_data.labels
        ),
        "Случайная": random_initialization(train_data.features, seed=42),
    }
    initial_margins = {
        name: calculate_margins(train_data.features, train_data.labels, weights)
        for name, weights in initial_weights.items()
    }
    final_margins = {
        name: calculate_margins(dataset.features, dataset.labels, best_weights)
        for name, dataset in datasets.items()
    }

    return EvaluationResults(
        best_run=best_name,
        validation_runs=validation_runs,
        validation_summary=validation_summary,
        learning_histories=learning_histories,
        test_scores=test_scores,
        test_metrics=evaluate_models(test_data.labels, test_scores),
        initial_margins=initial_margins,
        final_margins=final_margins,
        margin_summary=summarize_margins(final_margins),
    )
