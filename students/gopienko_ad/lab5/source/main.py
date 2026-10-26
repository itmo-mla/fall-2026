from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from students.gopienko_ad.lab5.source.dataset import StandardScaler, calculate_impute_medians, impute_zero_values, \
    train_val_test_split, analyze_dataset, load_dataset, COLUMNS
from students.gopienko_ad.lab5.source.logistic_regression import LogisticRegressionNewton, LogisticRegressionIRLS
from students.gopienko_ad.lab5.source.metrics import calculate_metrics
from students.gopienko_ad.lab5.source.plots import plot_loss, plot_confusion, plot_roc, plot_coefficients, \
    plot_probability_hist, plot_pca_2d_boundary

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result"
DATA_DIR = RESULT_DIR / "data"
STATISTICS_DIR = RESULT_DIR / "statistics"
PLOTS_DIR = RESULT_DIR / "plots"

FEATURES = COLUMNS[:-1]
RANDOM_STATE = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
MAX_ITER = 100
TOL = 1e-9
ALPHA = 0.0


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    result = calculate_metrics(model, X, y)
    eta = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    result["log_loss"] = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    return result


def compare_models(models: dict, X_train: np.ndarray, X_test: np.ndarray) -> dict[str, float | int]:
    newton = models["newton"]
    irls = models["irls"]
    reference = models["sklearn"]

    def parameters(model) -> np.ndarray:
        return np.r_[float(np.asarray(model.intercept_).item()),
                     np.asarray(model.coef_, dtype=float).ravel()]

    def proba(model, X) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    result: dict[str, float | int] = {
        "design_rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(X_train)), X_train]))),
        "design_columns": int(X_train.shape[1] + 1),
    }

    for left_name, right_name in (("newton", "irls"), ("newton", "sklearn"), ("irls", "sklearn")):
        left, right = models[left_name], models[right_name]
        prefix = f"{left_name}_vs_{right_name}"
        result[f"{prefix}_coeff_max_abs_diff"] = float(np.max(np.abs(parameters(left) - parameters(right))))
        result[f"{prefix}_train_proba_max_abs_diff"] = float(np.max(np.abs(proba(left, X_train) - proba(right, X_train))))
        result[f"{prefix}_test_proba_max_abs_diff"] = float(np.max(np.abs(proba(left, X_test) - proba(right, X_test))))
        result[f"{prefix}_test_class_disagreements"] = int(np.count_nonzero(left.predict(X_test) != right.predict(X_test)))
    return result


def main() -> None:
    for directory in (DATA_DIR, STATISTICS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    df = load_dataset(output_dir=DATA_DIR)
    analyze_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df,
        target_column="class",
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        stratify=True,
        random_state=RANDOM_STATE
    )

    print(f"\nРазмеры выборок: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    medians = calculate_impute_medians(X_train)
    X_train = impute_zero_values(X_train, medians).to_numpy(dtype=float)
    X_val = impute_zero_values(X_val, medians).to_numpy(dtype=float)
    X_test = impute_zero_values(X_test, medians).to_numpy(dtype=float)
    pd.DataFrame([medians]).to_csv(STATISTICS_DIR / "imputation_medians.csv", index=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    sklearn_settings = (
        {"penalty": None}
        if ALPHA == 0
        else {"C": 1.0 / (ALPHA * len(y_train))}
    )

    models = {
        "newton": LogisticRegressionNewton(alpha=ALPHA, tol=TOL, max_iter=MAX_ITER).fit(X_train, y_train),
        "irls": LogisticRegressionIRLS(alpha=ALPHA, tol=TOL, max_iter=MAX_ITER).fit(X_train, y_train),
        "sklearn": LogisticRegression(
            solver="newton-cholesky",
            tol=1e-11,
            max_iter=1000,
            **sklearn_settings
        ).fit(X_train, y_train)
    }

    print("\nСходимость собственных моделей:")
    for name in ("newton", "irls"):
        model = models[name]
        print(f"  {name}: итерации={model.n_iter_}, сходимость={model.converged_}, "
              f"loss={model.loss_history_[-1]:.8f}")
        if not model.converged_:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: {name} не достиг критерия остановки")

    rows = []
    for split, X, y in (("train", X_train, y_train),
                        ("val", X_val, y_val),
                        ("test", X_test, y_test)):
        for name, model in models.items():
            rows.append({"split": split, "model": name, **evaluate_model(model, X, y)})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(STATISTICS_DIR / "classification_metrics.csv", index=False)
    print("\nМетрики на тестовой выборке (порог 0.5):")
    print(metrics_df.loc[metrics_df["split"] == "test",
                         ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]]
          .to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    checks = compare_models(models, X_train=X_train, X_test=X_test)
    pd.DataFrame([checks]).to_csv(STATISTICS_DIR / "model_comparison.csv", index=False)
    print("\nСравнение моделей:")

    for name, value in checks.items():
        print(f"  {name}: {value:.3e}" if isinstance(value, float) else f"  {name}: {value}")
    if checks["design_rank"] < checks["design_columns"]:
        print("ВНИМАНИЕ: столбцы матрицы признаков линейно зависимы; "
              "коэффициенты могут отличаться даже при совпадающих вероятностях.")

    coefficients = {"feature": ["intercept", *FEATURES]}
    for name, model in models.items():
        coefficients[name] = [float(np.asarray(model.intercept_).item()),
                              *np.asarray(model.coef_, dtype=float).ravel().tolist()]
    pd.DataFrame(coefficients).to_csv(STATISTICS_DIR / "model_coefficients.csv", index=False)

    history_rows = []
    for name in ("newton", "irls"):
        model = models[name]
        for iteration, (loss, grad, step) in enumerate(zip(
            model.loss_history_, model.grad_norm_history_, model.step_history_
        )):
            history_rows.append({
                "model": name,
                "iteration": iteration,
                "train_objective": loss,
                "gradient_l2_norm": grad,
                "step_size": step,
            })

    pd.DataFrame(history_rows).to_csv(STATISTICS_DIR / "optimization_history.csv", index=False)

    test_predictions = pd.DataFrame({"y_true": y_test})
    for name, model in models.items():
        test_predictions[f"p1_{name}"] = model.predict_proba(X_test)[:, 1]
        test_predictions[f"pred_{name}"] = model.predict(X_test)
    test_predictions.to_csv(STATISTICS_DIR / "test_predictions.csv", index=False)

    plot_loss(models, PLOTS_DIR / "optimization_loss.png")
    plot_roc(models, X_test, y_test, PLOTS_DIR / "roc_curves.png")
    plot_confusion(models, X_test, y_test, PLOTS_DIR / "confusion_matrices.png")
    plot_coefficients(models, FEATURES, PLOTS_DIR / "coefficients.png")
    plot_probability_hist(models, X_test, y_test, PLOTS_DIR / "probabilities.png")
    plot_pca_2d_boundary(X_train, y_train, PLOTS_DIR / "pca_2d_boundary.png")


if __name__ == "__main__":
    main()
