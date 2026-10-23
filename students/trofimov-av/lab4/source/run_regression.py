from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data import load_data, prepare
from pca import PCA, effective_dimension


def predict_linear(x_train, y_train, x_test):
    train = np.column_stack((np.ones(len(x_train)), x_train))
    test = np.column_stack((np.ones(len(x_test)), x_test))
    weights = np.linalg.lstsq(train, y_train, rcond=None)[0]
    return test @ weights


def main():
    root = Path(__file__).resolve().parents[1]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    data, labels, _ = load_data()
    x, y = data[:, :3], data[:, 3]
    rows = []

    for seed in (0, 1, 2, 3, 42):
        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=seed, stratify=labels
        )
        for standardize in (False, True):
            mode = "Стандартизация" if standardize else "Центрирование"
            x_train, x_test, _, _ = prepare(x_train_raw, x_test_raw, standardize)
            full = PCA(3).fit(x_train)
            dimension = effective_dimension(full.explained_variance_ratio_, 0.95)
            baseline = predict_linear(x_train, y_train, x_test)
            for k in range(4):
                if k == 0:
                    prediction = baseline
                    reference_prediction = baseline
                    retained = 1.0
                else:
                    model = PCA(k).fit(x_train)
                    reference = SklearnPCA(n_components=k, svd_solver="full").fit(x_train)
                    prediction = predict_linear(model.transform(x_train), y_train, model.transform(x_test))
                    reference_prediction = predict_linear(
                        reference.transform(x_train), y_train, reference.transform(x_test)
                    )
                    retained = model.explained_variance_ratio_.sum()
                    np.testing.assert_allclose(prediction, reference_prediction, atol=1e-10)
                    if k == 3:
                        np.testing.assert_allclose(prediction, baseline, atol=1e-10)
                rows.append({
                    "seed": seed,
                    "preprocessing": mode,
                    "n_components": k,
                    "dimension_95": dimension,
                    "retained_variance": retained,
                    "test_mse": mean_squared_error(y_test, prediction),
                    "test_r2": r2_score(y_test, prediction),
                    "reference_mse": mean_squared_error(y_test, reference_prediction),
                    "prediction_max_error": np.max(np.abs(prediction - reference_prediction)),
                })

    results = pd.DataFrame(rows)
    summary = results.groupby(["preprocessing", "n_components"], sort=False).agg(
        mse_mean=("test_mse", "mean"),
        mse_std=("test_mse", "std"),
        r2_mean=("test_r2", "mean"),
        r2_std=("test_r2", "std"),
        prediction_max_error=("prediction_max_error", "max"),
    ).reset_index()
    results.to_csv(root / "results" / "regression.tsv", sep="\t", index=False)
    summary.to_csv(root / "results" / "regression_summary.tsv", sep="\t", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mode, group in summary.groupby("preprocessing", sort=False):
        for ax, metric, label in zip(axes, ("mse", "r2"), ("MSE, см²", "R²")):
            ax.errorbar(
                group["n_components"], group[f"{metric}_mean"],
                yerr=group[f"{metric}_std"], marker="o", capsize=4, label=mode,
            )
            ax.set(xlabel="Число компонент PCA", ylabel=label, title="Среднее ± std по 5 разбиениям")
            ax.set_xticks(range(4), ["Без PCA", "1", "2", "3"])
            ax.grid(alpha=0.25)
            ax.legend()
    figure.tight_layout()
    figure.savefig(root / "figures" / "regression.png", dpi=160)
    plt.close(figure)
    print(summary.to_string(index=False))
    print("Размерность при 95% по разбиениям:")
    print(results[results.n_components == 1][["seed", "preprocessing", "dimension_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
