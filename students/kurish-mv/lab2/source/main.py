import sys

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from crosscheck import cross_validate_variants
from dataset import prepare_wine
from figures import (
    plot_pca_prototypes,
    plot_pruning_trace,
    plot_quality_panels,
    plot_risk_curves,
)
from kernel_knn import WindowVoteKNN
from loo import loo_error_curve, pick_best_k, sklearn_loo_error_curve
from pruning import condense_training_set
from quality import (
    MODEL_LABELS,
    prediction_changes,
    prediction_time_ms,
    quality_metrics,
)

K_VALUES = np.arange(1, 41)
MIN_CLASS_FRACTION = 0.4


def _prepare_paths() -> tuple[Path, Path, Path]:
    source_dir = Path(__file__).resolve().parent
    lab_dir = source_dir.parent
    images_dir = lab_dir / "images"
    results_dir = source_dir / "res"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return lab_dir, images_dir, results_dir


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    lab_dir, images_dir, results_dir = _prepare_paths()
    data = prepare_wine()
    X_train, y_train = data.X_train, data.y_train
    X_test, y_test = data.X_test, data.y_test
    n_train = len(y_train)
    n_classes = len(data.class_names)

    print(f"wine: {len(data.frame)} объектов, {len(data.feature_names)} признаков, "
          f"классов: {n_classes}")
    print(f"train/test: {n_train}/{len(y_test)}; классы train: "
          f"{data.train_counts.tolist()}, test: {data.test_counts.tolist()}")

    self_full = np.arange(n_train)
    full_risks = loo_error_curve(
        K_VALUES, X_train, y_train, self_columns=self_full
    )
    full_k = pick_best_k(K_VALUES, full_risks)

    sklearn_risks = sklearn_loo_error_curve(K_VALUES, X_train, y_train)
    sklearn_k = pick_best_k(K_VALUES, sklearn_risks)

    selection = condense_training_set(
        X_train, y_train, min_class_fraction=MIN_CLASS_FRACTION, tolerance=0.0
    )
    keep_indices = selection.keep_indices
    n_kept = len(keep_indices)

    reduced_k_values = K_VALUES[K_VALUES <= min(K_VALUES.max(), n_kept - 2)]
    self_reduced = np.full(n_train, -1, dtype=int)
    self_reduced[keep_indices] = np.arange(n_kept)
    reduced_risks = loo_error_curve(
        reduced_k_values,
        X_train[keep_indices],
        y_train[keep_indices],
        X_train,
        y_train,
        self_columns=self_reduced,
    )
    reduced_k = pick_best_k(reduced_k_values, reduced_risks)

    small = condense_training_set(
        X_train, y_train, min_class_fraction=0.02, tolerance=0.0
    )
    sensitivity = {
        "keep_count": len(small.keep_indices),
        "risk": small.final_risk,
        "counts": np.bincount(
            y_train[small.keep_mask], minlength=n_classes
        ),
    }

    models = {
        "full": WindowVoteKNN(full_k).fit(X_train, y_train),
        "sklearn": KNeighborsClassifier(
            n_neighbors=sklearn_k, weights="uniform"
        ).fit(X_train, y_train),
        "reduced": WindowVoteKNN(reduced_k).fit(
            X_train[keep_indices], y_train[keep_indices]
        ),
    }
    predictions = {key: model.predict(X_test) for key, model in models.items()}

    risk_by_key = {
        "full": float(full_risks.min()),
        "sklearn": float(sklearn_risks.min()),
        "reduced": float(reduced_risks.min()),
    }
    objects_by_key = {
        "full": n_train,
        "sklearn": n_train,
        "reduced": n_kept,
    }
    chosen_k = {"full": full_k, "sklearn": sklearn_k, "reduced": reduced_k}

    rows = []
    for key, model in models.items():
        metrics = quality_metrics(y_test, predictions[key])
        rows.append({
            "key": key,
            "model": MODEL_LABELS[key],
            "k": chosen_k[key],
            "objects": objects_by_key[key],
            "loo_risk": risk_by_key[key],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "errors": metrics["errors"],
            "confusion": metrics["confusion"],
            "prediction_ms": prediction_time_ms(model, X_test),
        })
    metrics_table = pd.DataFrame(rows).set_index("key")

    X_raw = data.frame[data.feature_names].to_numpy(dtype=float)
    y_raw = data.frame["target"].to_numpy(dtype=int)
    cv_results = cross_validate_variants(
        X_raw, y_raw, K_VALUES, min_class_fraction=MIN_CLASS_FRACTION
    )

    curves = {
        "full": (K_VALUES, full_risks),
        "sklearn": (K_VALUES, sklearn_risks),
        "reduced": (reduced_k_values, reduced_risks),
    }
    best_k = {"full": full_k, "sklearn": sklearn_k, "reduced": reduced_k}

    plot_risk_curves(curves, best_k, images_dir / "01_loo_risk.png")
    plot_pruning_trace(
        selection.history, y_train, selection.keep_mask,
        data.class_names, images_dir / "02_pruning.png",
    )
    plot_pca_prototypes(
        X_train, y_train, selection.keep_mask,
        data.class_names, images_dir / "03_pca_prototypes.png",
    )
    plot_quality_panels(
        y_test, predictions, data.class_names, metrics_table,
        images_dir / "04_quality.png",
    )

    curve_table = pd.DataFrame(
        {"parzen_full": full_risks, "sklearn": sklearn_risks},
        index=pd.Index(K_VALUES, name="k"),
    )
    curve_table["parzen_reduced"] = pd.Series(
        reduced_risks, index=reduced_k_values
    ).reindex(K_VALUES)
    curve_table.to_csv(results_dir / "loo_curves.csv", encoding="utf-8-sig")

    metrics_table.drop(columns="confusion").to_csv(
        results_dir / "model_metrics.csv", encoding="utf-8-sig"
    )
    selection.history.to_csv(
        results_dir / "pruning_history.csv", index=False, encoding="utf-8-sig"
    )
    cv_results.to_csv(
        results_dir / "cross_validation.csv", index=False, encoding="utf-8-sig"
    )

    changes = prediction_changes(
        y_test, predictions["full"], predictions["reduced"]
    )
    disagreements = int(np.count_nonzero(
        predictions["full"] != predictions["sklearn"]
    ))

    print(f"\nПолная модель: k={full_k}, LOO={100 * risk_by_key['full']:.2f}%")
    print(f"sklearn KNN: k={sklearn_k}, LOO={100 * risk_by_key['sklearn']:.2f}%")
    print(f"Модель с эталонами: k={reduced_k}, "
          f"LOO={100 * risk_by_key['reduced']:.2f}%")
    print(f"Эталоны: {n_kept}/{n_train} ({100 * n_kept / n_train:.1f}%), "
          f"по классам {np.bincount(y_train[selection.keep_mask], minlength=n_classes).tolist()}")
    print(f"Без квоты: {sensitivity['keep_count']} эталонов, "
          f"риск 1NN={100 * sensitivity['risk']:.2f}%")
    print("\n" + metrics_table.drop(columns="confusion").round(4).to_string())
    print(f"Расхождений с sklearn на тесте: {disagreements}")
    print(f"Исправлено после отбора: {changes['corrected']}, "
          f"внесено ошибок: {changes['introduced']}")
    print(f"\nГрафики: {images_dir}")


if __name__ == "__main__":
    main()
