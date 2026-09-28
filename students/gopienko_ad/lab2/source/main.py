import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from students.gopienko_ad.lab2.source.dataset import (
    COLUMNS,
    COLUMNS_TO_IMPUTE,
    StandardScaler,
    analyze_dataset,
    load_dataset,
    train_val_test_split,
)
from students.gopienko_ad.lab2.source.knn_model import (
    GreedyPrototypeSelector,
    KNNClassifier,
    select_best_k_loo,
)
from students.gopienko_ad.lab2.source.metrics import (
    calculate_metrics,
    confusion_matrix,
)
from students.gopienko_ad.lab2.source.plots import (
    plot_loo_risk,
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_prototypes_2d,
)


RESULTS_DIR = Path("results")
DATA_DIR = RESULTS_DIR / "data"
STATISTICS_DIR = RESULTS_DIR / "statistics"
PLOTS_DIR = RESULTS_DIR / "plots"

K_VALUES = np.arange(1, 31)

PROTOTYPE_TOLERANCE = 0.0
MIN_PROTOTYPE_FRACTION = 0.40
MIN_CLASS_FRACTION = 0.50


def create_result_directories() -> None:
    for directory in (
        RESULTS_DIR,
        DATA_DIR,
        STATISTICS_DIR,
        PLOTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def remove_obsolete_results() -> None:
    obsolete_paths = (
        STATISTICS_DIR / "summary.json",
        STATISTICS_DIR / "validation_metrics.csv",
        STATISTICS_DIR / "prototype_history.csv",
        PLOTS_DIR / "prototype_history.png"
    )

    for path in obsolete_paths:
        if path.exists():
            path.unlink()


def preprocess_data(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x_train = x_train.astype(float).copy()
    x_val = x_val.astype(float).copy()
    x_test = x_test.astype(float).copy()

    feature_columns = COLUMNS[:-1]

    columns_to_impute = [
        feature_columns.index(column)
        for column in COLUMNS_TO_IMPUTE
    ]

    for column_index in columns_to_impute:
        train_non_zero = x_train[x_train[:, column_index] != 0, column_index]

        median = np.median(train_non_zero)

        for x in (
            x_train,
            x_val,
            x_test,
        ):
            zero_mask = x[:, column_index] == 0

            x[zero_mask, column_index] = median

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return (
        x_train,
        x_val,
        x_test,
    )


def metrics_to_frame(
    metrics_by_model: dict[str, dict[str, float],]
) -> pd.DataFrame:
    frame = pd.DataFrame(metrics_by_model).T
    frame.index.name = "model"

    return frame


def save_confusion_matrices(
    models: dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, np.ndarray]:
    matrices = {}

    for model_name, model in models.items():
        prediction = model.predict(X_test)

        matrices[model_name] = confusion_matrix(y_test, prediction)

    ((STATISTICS_DIR / "confusion_matrices.json")
    .write_text(
        json.dumps(
            {
                model_name: matrix.tolist()
                for model_name, matrix in matrices.items()
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    ))

    return matrices


def save_class_distribution(
    y_prototypes: np.ndarray,
) -> None:
    classes, counts = np.unique(
        y_prototypes,
        return_counts=True,
    )

    distribution = pd.DataFrame(
        {
            "class": classes,
            "count": counts,
        }
    )

    distribution.to_csv(
        STATISTICS_DIR
        / "prototype_class_distribution.csv",
        index=False,
    )


def main() -> None:
    create_result_directories()
    remove_obsolete_results()

    df = load_dataset(output_dir=DATA_DIR)
    analyze_dataset(df)

    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    ) = train_val_test_split(
        df=df,
        target_column="class",
        val_size=0.15,
        test_size=0.15,
        stratify=True,
        random_state=42
    )

    (
        x_train,
        x_val,
        x_test
    ) = preprocess_data(
        x_train,
        x_val,
        x_test
    )

    print("\nDataset split:")
    print(f"Train: {x_train.shape}")
    print(f"Validation: {x_val.shape}")
    print(f"Test: {x_test.shape}")

    (
        best_k,
        k_values,
        loo_risks
    ) = select_best_k_loo(
        x=x_train,
        y=y_train,
        k_values=K_VALUES
    )

    best_loo_risk = float(loo_risks[np.argmin(loo_risks)])

    print("\nLOO parameter selection:")
    print(f"Best k: {best_k}")
    print(
        f"Best LOO risk: "
        f"{best_loo_risk:.6f}"
    )

    pd.DataFrame(
        {
            "k": k_values,
            "loo_risk": loo_risks,
        }
    ).to_csv(STATISTICS_DIR / "loo_risk.csv", index=False)

    plot_loo_risk(
        k_values=k_values,
        risks=loo_risks,
        best_k=best_k,
        path=PLOTS_DIR / "loo_risk.png",
    )

    parzen_knn = KNNClassifier(k=best_k)
    parzen_knn.fit(x_train, y_train,)

    sklearn_knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
        weights="uniform",
    )

    sklearn_knn.fit(x_train, y_train,)

    min_prototypes = max(
        len(np.unique(y_train)),
        int(np.ceil(len(x_train) * MIN_PROTOTYPE_FRACTION)),
    )

    _, class_counts = np.unique(y_train, return_counts=True)

    min_per_class = max(1, int(np.ceil(np.min(class_counts) * MIN_CLASS_FRACTION)))

    selector = GreedyPrototypeSelector(
        tolerance=PROTOTYPE_TOLERANCE,
        min_per_class=min_per_class,
        min_prototypes=min_prototypes,
        max_removals=None
    )


    x_prototypes, y_prototypes = selector.fit_transform(x_train, y_train)
    prototype_classes, prototype_counts = np.unique(y_prototypes, return_counts=True)

    print("\nPrototype selection:")
    print(f"Before: {len(x_train)}")
    print(f"After: {len(x_prototypes)}")
    print(
        "Reduction: "
        f"{1.0 - len(x_prototypes) / len(x_train):.2%}"
    )

    print("\nPrototype class distribution:")

    for class_label, count in zip(
        prototype_classes,
        prototype_counts,
    ):
        print(f"class {class_label}: {count}")

    save_class_distribution(y_prototypes)

    plot_prototypes_2d(
        X=x_train,
        y=y_train,
        prototype_indices=selector.prototype_indices_,
        path=PLOTS_DIR / "prototypes_pca.png"
    )

    parzen_with_prototypes = KNNClassifier(k=best_k)
    parzen_with_prototypes.fit(x_prototypes, y_prototypes)

    models = {
        "Parzen KNN": parzen_knn,
        "sklearn KNN": sklearn_knn,
        "Parzen KNN + prototypes": parzen_with_prototypes
    }

    test_metrics = {
        model_name: calculate_metrics(model, x_test, y_test)
        for model_name, model in models.items()
    }

    metrics_frame = metrics_to_frame(test_metrics)

    print("\nTest metrics:")
    print(metrics_frame.round(4))

    metrics_frame.to_csv(STATISTICS_DIR / "metrics.csv")

    confusion_matrices = save_confusion_matrices(
        models=models,
        X_test=x_test,
        y_test=y_test
    )

    plot_confusion_matrices(
        matrices=confusion_matrices,
        path=PLOTS_DIR / "confusion_matrices.png"
    )

    plot_metric_comparison(
        metrics_by_model=test_metrics,
        path=PLOTS_DIR / "metrics_comparison.png"
    )

    print("\nAll results were saved to:")
    print(RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
