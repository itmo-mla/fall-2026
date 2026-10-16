from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from data_preparation import prepare_data
from knn import ParzenKNN, loo_risk_curve, pairwise_distances, parzen_weights
from reference_selection import (
    ccv,
    ccv_weights,
    compactness_profile,
    greedy_add_references,
)
from visualization import (
    plot_compactness_profile,
    plot_confusion_matrices,
    plot_loo_risk,
    plot_model_comparison,
    plot_references,
    plot_selection_history,
)


MAX_K = 50
# Control length for CCV: the same 20% share as the train/test split.
CONTROL_SHARE = 0.2


class NearestNeighbor:
    """1NN classifier: the class of the single nearest training object."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestNeighbor":
        self.X_, self.y_ = np.asarray(X, dtype=float), np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.y_[np.argmin(pairwise_distances(np.asarray(X, dtype=float), self.X_), axis=1)]


def main() -> None:
    data = prepare_data()
    X_train, y_train = data.X_train, data.y_train
    output_dir = Path(__file__).parents[1] / "graphs"
    output_dir.mkdir(exist_ok=True)

    # 1. Choice of k by LOO.
    k_values = range(1, min(MAX_K, len(X_train) - 2) + 1)
    risks = loo_risk_curve(X_train, y_train, k_values)
    best_k = min(risks, key=risks.get)
    plot_loo_risk(risks, best_k, output_dir / "loo_empirical_risk.png")

    # 2. Parzen window classifier and the sklearn reference implementation.
    custom = ParzenKNN(k=best_k).fit(X_train, y_train)
    # All training objects are neighbors and get the same weights as in ParzenKNN.
    sklearn_parzen = KNeighborsClassifier(
        n_neighbors=len(X_train), weights=partial(parzen_weights, k=best_k)
    ).fit(X_train, y_train)
    sklearn_distance = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    sklearn_distance.fit(X_train, y_train)

    # 3. Reference selection for 1NN by greedy CCV minimization.
    control_size = round(CONTROL_SHARE * len(X_train))
    weights = ccv_weights(len(X_train), control_size)
    selection = greedy_add_references(X_train, y_train, control_size)
    references = selection.references
    nearest_full = NearestNeighbor().fit(X_train, y_train)
    nearest_references = NearestNeighbor().fit(X_train[references], y_train[references])
    parzen_references = ParzenKNN(k=min(best_k, len(references) - 1)).fit(
        X_train[references], y_train[references]
    )

    profile_full = compactness_profile(X_train, y_train, len(weights))
    profile_references = compactness_profile(X_train, y_train, len(weights), references)
    full_ccv = ccv(profile_full, weights)
    plot_compactness_profile(
        profile_full, profile_references, weights, output_dir / "compactness_profile.png"
    )

    start_size = len(np.unique(y_train))
    test_errors = [
        float(np.mean(
            NearestNeighbor().fit(X_train[references[:size]], y_train[references[:size]])
            .predict(data.X_test) != data.y_test
        ))
        for size in range(start_size, len(references) + 1)
    ]
    full_test_error = float(np.mean(nearest_full.predict(data.X_test) != data.y_test))
    plot_selection_history(
        selection.ccv_history,
        test_errors,
        start_size,
        full_ccv,
        full_test_error,
        output_dir / "selection_history.png",
    )

    # Objects that 1NN on the references still misclassifies are marked as noise.
    is_reference = np.zeros(len(y_train), dtype=bool)
    is_reference[references] = True
    noise = np.flatnonzero((nearest_references.predict(X_train) != y_train) & ~is_reference)
    plot_references(X_train, y_train, references, noise, output_dir / "selected_references.png")

    # 4. Comparison on the test set.
    models = {
        "Parzen-KNN": (custom, best_k, len(X_train)),
        "sklearn (окно Парзена)": (sklearn_parzen, best_k, len(X_train)),
        "sklearn KNN (1/d)": (sklearn_distance, best_k, len(X_train)),
        "1NN": (nearest_full, 1, len(X_train)),
        "1NN + эталоны": (nearest_references, 1, len(references)),
        "Parzen-KNN + эталоны": (parzen_references, parzen_references.k, len(references)),
    }
    predictions = {name: model.predict(data.X_test) for name, (model, _, _) in models.items()}
    results = pd.DataFrame(
        [
            {
                "model": name,
                "k": k,
                "accuracy": accuracy_score(data.y_test, predictions[name]),
                "f1": f1_score(data.y_test, predictions[name]),
                "train_objects": train_objects,
            }
            for name, (_, k, train_objects) in models.items()
        ]
    )
    plot_model_comparison(results, output_dir / "model_comparison.png")
    plot_confusion_matrices(data.y_test, predictions, output_dir / "confusion_matrices.png")
    results.to_csv(output_dir / "comparison_metrics.csv", index=False)
    pd.DataFrame(
        {"k": list(risks), "loo_error": list(risks.values())}
    ).to_csv(output_dir / "loo_risks.csv", index=False)

    agreement = np.mean(predictions["Parzen-KNN"] == predictions["sklearn (окно Парзена)"])
    print(f"LOO selected k: {best_k}; risk: {risks[best_k]:.4f}")
    print(f"Parzen-KNN vs sklearn (same weights) agreement: {agreement:.1%}")
    print(
        f"CCV (control {control_size} of {len(X_train)}): full sample {full_ccv:.4f}, "
        f"references {selection.ccv_history[-1]:.4f}"
    )
    print(
        f"References: {len(references)} / {len(X_train)} ({len(references) / len(X_train):.1%}), "
        f"noise objects: {len(noise)}"
    )
    print(results.to_string(index=False))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
