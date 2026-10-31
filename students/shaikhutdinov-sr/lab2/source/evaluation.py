from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from experiments import ExperimentResults


def evaluate_models(
    results: ExperimentResults,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    predictions = {}
    metric_rows = []
    for experiment in (results.full, results.reference, results.reduced):
        predicted_labels = experiment.classifier.predict(x_test)
        predictions[experiment.name] = predicted_labels

        prediction_times = []
        for _ in range(7):
            start = perf_counter()
            experiment.classifier.predict(x_test)
            prediction_times.append(perf_counter() - start)
        prediction_ms = float(np.median(prediction_times)) * 1000

        selected_k = int(experiment.loo_curve.idxmin())
        loo_risk = float(experiment.loo_curve.min())
        accuracy = accuracy_score(y_test, predicted_labels)
        f1_macro = f1_score(y_test, predicted_labels, average="macro")
        errors = int(np.count_nonzero(predicted_labels != y_test))
        metric_rows.append({
            "model": experiment.name,
            "k": selected_k,
            "neighbor_count": experiment.neighbor_count,
            "loo_risk": loo_risk,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "errors": errors,
            "prediction_ms": prediction_ms,
        })

    metrics = pd.DataFrame(metric_rows)
    metrics = metrics.set_index("model")
    return metrics, predictions
