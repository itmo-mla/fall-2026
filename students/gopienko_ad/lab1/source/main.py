from pathlib import Path

import numpy as np
import pandas as pd

from dataset import (
    load_dataset,
    analyze_dataset,
    impute_zero_values, COLUMNS
)
from students.gopienko_ad.lab1.source.dataset import StandardScaler, train_val_test_split
from students.gopienko_ad.lab1.source.linear_classifier import LinearClassifier, RidgeClassifier
from students.gopienko_ad.lab1.source.metrics import calculate_metrics
from students.gopienko_ad.lab1.source.plots import plot_feature_correlations, plot_objective, plot_recurrent_quality, \
    plot_multistart_objectives, plot_confusion_matrix, plot_margins, plot_full_correlation_heatmap


def train_multistart(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        l2: float,
        epochs: int,
        runs: int,
        learning_rate: float,
        momentum: float,
        lr_decay: float,
        recurrent_alpha: float,
        presentation: str = "random",
):
    best_model = None
    best_seed = None
    best_objective = np.inf

    rows = []

    for seed in range(runs):
        model = LinearClassifier(
            l2=l2,
            seed=seed,
        )

        model.fit_sgd_momentum(
            X_train,
            y_train,
            epochs=epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            lr_decay=lr_decay,
            recurrent_alpha=recurrent_alpha,
            init="random",
            presentation=presentation,
            seed=seed,
        )

        val_objective = model.objective(X_val, y_val)

        rows.append(
            {
                "seed": seed,
                "validation_objective": val_objective,
            }
        )

        if val_objective < best_objective:
            best_objective = val_objective
            best_model = model
            best_seed = seed

    multistart_table = pd.DataFrame(rows)

    return (
        best_model,
        best_seed,
        multistart_table,
    )


def evaluate_models(
        models: dict,
        X_val: np.ndarray,
        y_val: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for name, model in models.items():
        row = {
            "model": name,
            "validation_objective": model.objective(
                X_val,
                y_val,
            ),
        }

        metrics = calculate_metrics(model, X_val, y_val)
        row.update(metrics)
        rows.append(row)

    validation_table = pd.DataFrame(rows)

    validation_table = validation_table.sort_values(
        by=[
            "f1",
            "roc_auc",
            "accuracy",
        ],
        ascending=False,
    )

    return validation_table


def main():
    random_state = 42
    l2 = 1e-3
    epochs = 160
    multistart_runs = 12
    learning_rate = 0.003
    momentum = 0.9
    lr_decay = 2e-4
    recurrent_alpha = 0.02

    results_dir = Path(
        "results"
    )

    plots_dir = results_dir / "plots"

    results_dir.mkdir(parents=True, exist_ok=True,)
    plots_dir.mkdir(parents=True, exist_ok=True,)

    feature_names = np.asarray(COLUMNS[:-1])

    df = load_dataset()
    analyze_dataset(df)
    df = impute_zero_values(df)

    df["class"] = np.where(df["class"] == 1,1.0, -1.0)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = train_val_test_split(
        df,
        test_size=0.15,
        val_size=0.15,
        random_state=random_state,
        stratify=True,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    sgd_correlation_random = LinearClassifier(l2=l2, seed=random_state,)

    correlations = (
        sgd_correlation_random
        .correlation_initialization(
            X_train,
            y_train,
        )
    )

    sgd_correlation_random.fit_sgd_momentum(
        X_train,
        y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        lr_decay=lr_decay,
        recurrent_alpha=recurrent_alpha,
        init="correlation",
        presentation="random",
        seed=random_state,
    )

    (
        sgd_multistart_random,
        best_seed,
        multistart_table,
    ) = train_multistart(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        l2=l2,
        epochs=epochs,
        runs=multistart_runs,
        learning_rate=learning_rate,
        momentum=momentum,
        lr_decay=lr_decay,
        recurrent_alpha=recurrent_alpha,
        presentation="random",
    )

    sgd_correlation_margin = LinearClassifier(
        l2=l2,
        seed=random_state,
    )

    sgd_correlation_margin.fit_sgd_momentum(
        X_train,
        y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        lr_decay=lr_decay,
        recurrent_alpha=recurrent_alpha,
        init="correlation",
        presentation="margin",
        seed=random_state,
    )

    steepest = LinearClassifier(l2=l2, seed=random_state,)

    steepest.fit_steepest_descent(
        X_train,
        y_train,
        max_iterations=300,
        tolerance=1e-10,
        init="correlation",
        seed=random_state,
    )

    ridge = RidgeClassifier(l2=l2,)
    ridge.fit(X_train, y_train)

    models = {
        "SGD correlation + random":
            sgd_correlation_random,

        "SGD multistart + random":
            sgd_multistart_random,

        "SGD correlation + |margin|":
            sgd_correlation_margin,

        "Steepest descent":
            steepest,

        "Ridge analytical":
            ridge,
    }

    validation_table = evaluate_models(
        models,
        X_val,
        y_val,
    )

    validation_table.to_csv(
        results_dir
        / "validation_metrics.csv",
        index=False,
    )

    multistart_table.to_csv(
        results_dir
        / "multistart.csv",
        index=False,
    )

    candidate_names = [
        "SGD correlation + random",
        "SGD multistart + random",
        "SGD correlation + |margin|",
        "Steepest descent",
    ]

    candidate_table = validation_table[
        validation_table[
            "model"
        ].isin(
            candidate_names
        )
    ]

    best_name = str(
        candidate_table.iloc[0][
            "model"
        ]
    )

    best_model = models[best_name]

    test_metrics = calculate_metrics(
        best_model,
        X_test,
        y_test,
    )

    ridge_test_metrics = calculate_metrics(
        ridge,
        X_test,
        y_test,
    )

    test_table = pd.DataFrame(
        [
            {
                "model": best_name,
                **test_metrics,
            },
            {
                "model": "Ridge analytical",
                **ridge_test_metrics,
            },
        ]
    )

    test_table.to_csv(
        results_dir
        / "test_metrics.csv",
        index=False,
    )

    reference_comparison = pd.DataFrame(
        [
            {
                "weights_l2_distance": float(
                    np.linalg.norm(
                        steepest.w
                        - ridge.w
                    )
                ),

                "bias_absolute_difference": float(
                    abs(
                        steepest.b
                        - ridge.b
                    )
                ),

                "train_objective_steepest":
                    steepest.objective(
                        X_train,
                        y_train,
                    ),

                "train_objective_ridge":
                    ridge.objective(
                        X_train,
                        y_train,
                    ),
            }
        ]
    )

    reference_comparison.to_csv(
        results_dir
        / "reference_comparison.csv",
        index=False,
    )

    plot_feature_correlations(
        feature_names,
        correlations,
        output_dir=plots_dir,
    )

    plot_full_correlation_heatmap(
        df,
        output_dir=plots_dir,
    )

    plot_margins(
        best_model,
        X_test,
        y_test,
        output_dir=plots_dir,
    )

    plot_objective(
        {
            name: model
            for name, model
            in models.items()
            if hasattr(
                model,
                "history",
            )
        },
        output_dir=plots_dir,
    )

    plot_recurrent_quality(
        sgd_correlation_random,
        output_dir=plots_dir,
    )

    plot_confusion_matrix(
        y_test,
        best_model.predict(
            X_test
        ),
        output_dir=plots_dir,
    )

    plot_multistart_objectives(
        multistart_table[
            "seed"
        ].to_numpy(),

        multistart_table[
            "validation_objective"
        ].to_numpy(),

        best_seed,
        output_dir=plots_dir,
    )

    print(
        "\nValidation results:"
    )

    print(validation_table.to_string(index=False))

    print("\nTest results:")
    print(test_table.to_string(index=False))

    print(
        f"\nBest model: "
        f"{best_name}"
    )

    print(
        f"Best multistart seed: "
        f"{best_seed}"
    )

    print(
        f"Results saved to: "
        f"{results_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
