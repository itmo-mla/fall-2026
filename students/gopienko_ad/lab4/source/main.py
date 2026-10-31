from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from students.gopienko_ad.lab4.source.dataset import load_dataset, analyze_dataset, train_val_test_split, StandardScaler
from students.gopienko_ad.lab4.source.metrics import regression_metrics, compare_pca, relative_reconstruction_error, \
    reconstruction_mse
from students.gopienko_ad.lab4.source.pca import PCACustom
from students.gopienko_ad.lab4.source.plots import plot_predictions, plot_validation_regression, \
    plot_reconstruction_errors, plot_pca_2d, plot_explained_variance
from students.gopienko_ad.lab4.source.regression import LinearRegressionSVD

RESULT_DIR = Path(__file__).resolve().parent / "result"
DATA_DIR = RESULT_DIR / "data"
STATISTICS_DIR = RESULT_DIR / "statistics"
PLOTS_DIR = RESULT_DIR / "plots"

RANDOM_STATE = 42
EPSILON = 0.05
ALPHA = 0.5

def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return regression_metrics(y, model.predict(X))

def model_metrics_row(
    k: int, name: str, model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict[str, float | int | str]:

    row: dict[str, float | int | str] = {"n_components": k, "model": name}
    for split, X, y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        row.update({f"{split}_{metric}": value for metric, value in evaluate_model(model, X, y).items()})

    return row

def main() -> None:
    for directory in (DATA_DIR, STATISTICS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    df = load_dataset(output_path=DATA_DIR / "diabetes_regression.csv")
    analyze_dataset(df, output_path=STATISTICS_DIR / "dataset_description.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target_column="target", val_size=0.15, test_size=0.15,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print(f"\nРазмеры выборок: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    custom_full = PCACustom().fit(X_train)
    sklearn_full = PCA(svd_solver="full").fit(X_train)
    full_comparison = compare_pca(custom_full, sklearn_full, X_train)
    print("\nСравнение полной PCA с sklearn:")
    for name, value in full_comparison.items():
        print(f"  {name}: {value:.3e}")

    k_effective = custom_full.effective_dimension(EPSILON)
    n_features = X_train.shape[1]
    ks = range(1, n_features + 1)
    cumulative_variance = np.cumsum(custom_full.all_explained_variance_ratio_)

    pd.DataFrame({
        "n_components": list(ks),
        "singular_value": custom_full.all_singular_values_,
        "eigenvalue": custom_full.all_explained_variance_,
        "explained_variance_ratio": custom_full.all_explained_variance_ratio_,
        "cumulative_explained_variance": cumulative_variance,
        "theoretical_reconstruction_error": 1.0 - cumulative_variance,
    }).to_csv(STATISTICS_DIR / "pca_spectrum.csv", index=False)

    reconstruction_rows = []
    comparison_rows = []
    regression_rows = []
    validation_rows = []
    test_predictions_by_k: dict[int, np.ndarray] = {}

    print("\nРезультаты при разном числе компонент:")
    print(f"{'k':>2} | {'explained %':>11} | {'E_train':>9} | {'val RMSE custom':>15} | {'val RMSE sklearn':>16}")
    for k in ks:
        pca = PCACustom(n_components=k).fit(X_train)
        pca_ref = PCA(n_components=k, svd_solver="full").fit(X_train)

        Z_train = pca.transform(X_train)
        Z_val = pca.transform(X_val)
        Z_test = pca.transform(X_test)
        Z_train_ref = pca_ref.transform(X_train)
        Z_val_ref = pca_ref.transform(X_val)
        Z_test_ref = pca_ref.transform(X_test)

        comparison_rows.append({
            "n_components": k,
            **compare_pca(pca, pca_ref, X_train),
        })

        X_train_rec = pca.inverse_transform(Z_train)
        X_val_rec = pca.inverse_transform(Z_val)
        X_test_rec = pca.inverse_transform(Z_test)
        reconstruction_rows.append({
            "n_components": k,
            "explained_variance_ratio_sum": float(cumulative_variance[k - 1]),
            "theoretical_E_m": custom_full.reconstruction_error_ratio(k),
            "train_E_m": relative_reconstruction_error(X_train, X_train_rec, pca.mean_),
            "val_E_m": relative_reconstruction_error(X_val, X_val_rec, pca.mean_),
            "test_E_m": relative_reconstruction_error(X_test, X_test_rec, pca.mean_),
            "train_reconstruction_mse": reconstruction_mse(X_train, X_train_rec),
            "val_reconstruction_mse": reconstruction_mse(X_val, X_val_rec),
            "test_reconstruction_mse": reconstruction_mse(X_test, X_test_rec),
        })

        custom_reg = LinearRegressionSVD(alpha=ALPHA).fit(Z_train, y_train)
        sklearn_reg = Ridge(alpha=ALPHA, solver="svd").fit(Z_train_ref, y_train)

        custom_row = model_metrics_row(
            k, "Custom PCA + Ridge SVD", custom_reg,
            Z_train, y_train, Z_val, y_val, Z_test, y_test,
        )
        sklearn_row = model_metrics_row(
            k, "sklearn PCA + Ridge", sklearn_reg,
            Z_train_ref, y_train, Z_val_ref, y_val, Z_test_ref, y_test,
        )

        regression_rows.extend((custom_row, sklearn_row))
        validation_rows.append({
            "n_components": k,
            "val_mae_custom": custom_row["val_mae"],
            "val_rmse_custom": custom_row["val_rmse"],
            "val_r2_custom": custom_row["val_r2"],
            "val_mae_sklearn": sklearn_row["val_mae"],
            "val_rmse_sklearn": sklearn_row["val_rmse"],
            "val_r2_sklearn": sklearn_row["val_r2"],
        })

        test_predictions_by_k[k] = custom_reg.predict(Z_test)
        print(f"{k:>2} | {cumulative_variance[k - 1] * 100:>10.2f}% | "
              f"{reconstruction_rows[-1]['train_E_m']:>9.5f} | "
              f"{custom_row['val_rmse']:>15.4f} | {sklearn_row['val_rmse']:>16.4f}")

    reconstruction_df = pd.DataFrame(reconstruction_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    regression_df = pd.DataFrame(regression_rows)
    validation_df = pd.DataFrame(validation_rows)
    reconstruction_df.to_csv(STATISTICS_DIR / "pca_reconstruction.csv", index=False)
    comparison_df.to_csv(STATISTICS_DIR / "pca_comparison.csv", index=False)
    regression_df.to_csv(STATISTICS_DIR / "regression_by_components.csv", index=False)
    validation_df.to_csv(STATISTICS_DIR / "regression_validation.csv", index=False)
    regression_df[["n_components", "model", "test_mae", "test_mse", "test_rmse", "test_r2"]].to_csv(
        STATISTICS_DIR / "regression_test_by_components.csv", index=False
    )

    k_best = int(validation_df.loc[validation_df["val_rmse_custom"].idxmin(), "n_components"])
    print(f"\nЛучшее количество компонент по validation RMSE: {k_best}")
    print(f"Эффективная размерность по дисперсии: {k_effective}")

    baseline_custom = LinearRegressionSVD(alpha=ALPHA).fit(X_train, y_train)
    baseline_sklearn = Ridge(alpha=ALPHA, solver="svd").fit(X_train, y_train)
    baseline_rows = [
        model_metrics_row(n_features, "Custom Ridge without PCA", baseline_custom,
                          X_train, y_train, X_val, y_val, X_test, y_test),
        model_metrics_row(n_features, "sklearn Ridge without PCA", baseline_sklearn,
                          X_train, y_train, X_val, y_val, X_test, y_test),
    ]
    pd.DataFrame(baseline_rows).to_csv(STATISTICS_DIR / "regression_baseline.csv", index=False)

    selected_rows = [baseline_rows[0]]
    selected_predictions = {"Ridge without PCA": baseline_custom.predict(X_test)}
    for k in dict.fromkeys((k_effective, k_best)):
        row = regression_df.loc[
            (regression_df["n_components"] == k)
            & (regression_df["model"] == "Custom PCA + Ridge SVD")
            ].iloc[0].to_dict()
        row["model"] = f"Custom PCA + Ridge (k={k})"
        selected_rows.append(row)
        selected_predictions[f"PCA k={k}"] = test_predictions_by_k[k]
    pd.DataFrame(selected_rows).to_csv(STATISTICS_DIR / "regression_selected_test.csv", index=False)

    print("\nИтоговое сравнение на test:")
    for row in selected_rows:
        print(f"{row['model']:<32} RMSE={row['test_rmse']:.4f}, "
              f"MAE={row['test_mae']:.4f}, R2={row['test_r2']:.4f}")

    plot_explained_variance(custom_full.all_explained_variance_ratio_, PLOTS_DIR / "explained_variance.png")
    plot_reconstruction_errors(reconstruction_df, PLOTS_DIR / "reconstruction_errors.png", epsilon=EPSILON)
    plot_pca_2d(custom_full.transform(X_train)[:, :2], y_train, PLOTS_DIR / "pca_2d.png")
    plot_validation_regression(validation_df, PLOTS_DIR / "validation_rmse.png", k_best)
    plot_predictions(y_test, selected_predictions, PLOTS_DIR / "test_predictions.png")



if __name__ == "__main__":
    main()
