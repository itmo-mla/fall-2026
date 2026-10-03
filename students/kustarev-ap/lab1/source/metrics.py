import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def pm1_to_01(y_pm1):
    return ((y_pm1 + 1) / 2).astype(int)


def evaluate_predictions(y_true_pm1, y_pred_pm1, scores=None):
    y_true = pm1_to_01(y_true_pm1)
    y_pred = pm1_to_01(y_pred_pm1)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if scores is not None:
        try:
            result["roc_auc"] = roc_auc_score(y_true, scores)
        except ValueError:
            result["roc_auc"] = np.nan
    return result


def evaluate_classifier(model, X, y_pm1):
    y_pred = model.predict(X)
    scores = model.decision_function(X) if hasattr(model, "decision_function") else None
    return evaluate_predictions(y_pm1, y_pred, scores)


def results_table(results):
    df = pd.DataFrame(results).T
    cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in df.columns]
    return df[cols].sort_values("roc_auc", ascending=False) if "roc_auc" in cols else df[cols]
