import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from dataset import scale_by_train
from kernel_knn import WindowVoteKNN
from loo import loo_error_curve, pick_best_k, sklearn_loo_error_curve
from pruning import condense_training_set
from quality import quality_metrics


def cross_validate_variants(
    X: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    min_class_fraction: float = 0.4,
) -> pd.DataFrame:
    """Стратифицированная CV: подбор k и отбор эталонов внутри каждого фолда."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    k_values = np.asarray(k_values, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        X_train, X_test = scale_by_train(X[train_index], X[test_index])
        y_train, y_test = y[train_index], y[test_index]
        n_train = len(y_train)

        full_risks = loo_error_curve(
            k_values, X_train, y_train, self_columns=np.arange(n_train)
        )
        full_k = pick_best_k(k_values, full_risks)

        selection = condense_training_set(
            X_train, y_train, min_class_fraction=min_class_fraction, tolerance=0.0
        )
        keep = selection.keep_indices
        reduced_k_values = k_values[k_values <= min(k_values.max(), len(keep) - 2)]
        self_columns = np.full(n_train, -1, dtype=int)
        self_columns[keep] = np.arange(len(keep))
        reduced_risks = loo_error_curve(
            reduced_k_values,
            X_train[keep],
            y_train[keep],
            X_train,
            y_train,
            self_columns=self_columns,
        )
        reduced_k = pick_best_k(reduced_k_values, reduced_risks)

        sklearn_risks = sklearn_loo_error_curve(k_values, X_train, y_train)
        sklearn_k = pick_best_k(k_values, sklearn_risks)

        models = {
            "full": WindowVoteKNN(full_k).fit(X_train, y_train),
            "sklearn": KNeighborsClassifier(n_neighbors=sklearn_k).fit(
                X_train, y_train
            ),
            "reduced": WindowVoteKNN(reduced_k).fit(
                X_train[keep], y_train[keep]
            ),
        }
        objects = {"full": n_train, "sklearn": n_train, "reduced": len(keep)}
        chosen_k = {"full": full_k, "sklearn": sklearn_k, "reduced": reduced_k}

        for key, model in models.items():
            metrics = quality_metrics(y_test, model.predict(X_test))
            rows.append({
                "fold": fold,
                "model": key,
                "k": chosen_k[key],
                "objects": objects[key],
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "errors": metrics["errors"],
            })

    return pd.DataFrame(rows)
