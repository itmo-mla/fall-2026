from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from data import load_data
from logistic import LogisticRegression


def main():
    root = Path(__file__).resolve().parents[1]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    rows = []
    for seed in (0, 1, 2, 3, 42):
        x_train, x_test, y_train, y_test = load_data(seed)
        for l2 in (0.01, 0.1, 1.0, 10.0):
            reference = SklearnLogisticRegression(C=1 / l2, solver="lbfgs", tol=1e-12, max_iter=2000)
            reference.fit(x_train, y_train)
            reference_probability = reference.predict_proba(x_test)[:, 1]
            reference_beta = np.r_[reference.intercept_, reference.coef_.ravel()]
            for method in ("newton", "irls", "sklearn"):
                if method == "sklearn":
                    model = reference
                    iterations = int(reference.n_iter_[0])
                    gradient_norm = np.nan
                    converged = iterations < reference.max_iter
                    beta = reference_beta
                else:
                    model = LogisticRegression(method=method, l2=l2).fit(x_train, y_train)
                    iterations = model.n_iter_
                    gradient_norm = model.gradient_history_[-1]
                    converged = model.converged_
                    beta = np.r_[model.intercept_, model.coef_]
                    np.testing.assert_allclose(model.predict_proba(x_test)[:, 1], reference_probability, atol=1e-6, rtol=0)
                    np.testing.assert_allclose(beta, reference_beta, atol=1e-5, rtol=0)
                probability = model.predict_proba(x_test)[:, 1]
                prediction = model.predict(x_test)
                assert converged, (seed, l2, method)
                rows.append({
                    "seed": seed, "l2": l2, "method": method,
                    "accuracy": accuracy_score(y_test, prediction),
                    "f1": f1_score(y_test, prediction),
                    "log_loss": log_loss(y_test, probability),
                    "roc_auc": roc_auc_score(y_test, probability),
                    "iterations": iterations, "gradient_norm": gradient_norm,
                    "converged": converged,
                    "probability_max_error": np.max(np.abs(probability - reference_probability)),
                    "coefficient_max_error": np.max(np.abs(beta - reference_beta)),
                    "prediction_agreement": np.mean(prediction == reference.predict(x_test)),
                    "coefficient_norm": np.linalg.norm(beta[1:]),
                })

    results = pd.DataFrame(rows)
    summary = results.groupby(["l2", "method"], sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), log_loss_mean=("log_loss", "mean"),
        log_loss_std=("log_loss", "std"), roc_auc_mean=("roc_auc", "mean"),
        iterations_mean=("iterations", "mean"), coefficient_norm_mean=("coefficient_norm", "mean"),
        probability_max_error=("probability_max_error", "max"),
        coefficient_max_error=("coefficient_max_error", "max"),
        prediction_agreement=("prediction_agreement", "min"),
    ).reset_index()
    results.to_csv(root / "results" / "experiments.tsv", sep="\t", index=False)
    summary.to_csv(root / "results" / "summary.tsv", sep="\t", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for method, group in summary.groupby("method", sort=False):
        for ax, metric in zip(axes[:2], ("accuracy", "log_loss")):
            ax.errorbar(group.l2, group[f"{metric}_mean"], yerr=group[f"{metric}_std"],
                        marker={"newton": "o", "irls": "x", "sklearn": "+"}[method],
                        markersize=9, capsize=4, label=method)
        axes[2].plot(group.l2, group.coefficient_norm_mean,
                     marker={"newton": "o", "irls": "x", "sklearn": "+"}[method],
                     markersize=9, label=method)
    for ax, ylabel in zip(axes, ("Test accuracy", "Test log-loss", "Норма коэффициентов")):
        ax.set(xscale="log", xlabel="λ: сила L2-регуляризации", ylabel=ylabel)
        ax.set_title("Среднее по 5 разбиениям")
        ax.grid(alpha=0.25)
        ax.legend()
    figure.tight_layout()
    figure.savefig(root / "figures" / "comparison.png", dpi=160)
    plt.close(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
