from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegressionCustom


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
    }, probabilities


def main():
    X_train, X_test, y_train, y_test = load_data()

    C = 1.0
    alpha = 1.0 / (len(X_train) * C)

    models = {
        "Newton-Raphson": LogisticRegressionCustom(
            method="newton",
            alpha=alpha,
            max_iter=100,
            tol=1e-10,
        ),
        "IRLS": LogisticRegressionCustom(
            method="irls",
            alpha=alpha,
            max_iter=100,
            tol=1e-10,
        ),
        "sklearn": LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-12,
        ),
    }

    metrics = []
    probabilities = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        result, proba = evaluate_model(
            name, model, X_test, y_test
        )

        metrics.append(result)
        probabilities[name] = proba

        print(f"\n{name}")
        for metric, value in result.items():
            if metric != "model":
                print(f"  {metric}: {value:.6f}")

    # Сравнение с эталонной реализацией
    reference = probabilities["sklearn"]

    for name in ("Newton-Raphson", "IRLS"):
        difference = np.max(
            np.abs(probabilities[name] - reference)
        )

        print(f"\n{name}: max probability difference = {difference:.10f}")

        assert difference < 1e-5, (
            f"{name} differs from sklearn: {difference}"
        )

    print("\nParity checks passed.")
    # Сохранение метрик
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False,
    )

    # ROC-кривые
    plt.figure(figsize=(8, 6))

    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)

        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # Сравнение предсказанных вероятностей
    plt.figure(figsize=(7, 6))

    for name in ("Newton-Raphson", "IRLS"):
        plt.scatter(
            reference,
            probabilities[name],
            alpha=0.5,
            s=15,
            label=name,
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("sklearn probability")
    plt.ylabel("Custom model probability")
    plt.title("Predicted probability comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "probability_comparison.png",
        dpi=150,
    )
    plt.close()

    print("\nResults saved to:", RESULTS_DIR)
    print("Images saved to:", IMAGES_DIR)


if __name__ == "__main__":
    main()