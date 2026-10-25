from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data import FEATURES, load_data
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import SVC
from svm import decision_function, fit_svm, predict

C = 1.0
GAMMA = 1.0
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_fake": f1_score(y_true, y_pred, pos_label=-1),
    }


def plot_dataset(X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Настоящие", alpha=0.7)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], label="Поддельные", alpha=0.7)
    ax.set(
        title="Выборка банкнот",
        xlabel="Верхнее поле, мм",
        ylabel="Длина, мм",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "dataset.png", dpi=180)
    plt.close(fig)


def plot_boundary(
    ax,
    score_function,
    X: np.ndarray,
    y: np.ndarray,
    scaler,
    title: str,
    support_vectors: np.ndarray,
) -> None:
    x_values = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 250)
    y_values = np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 250)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = score_function(scaler.transform(grid)).reshape(xx.shape)

    ax.contourf(xx, yy, scores, alpha=0.18, levels=[scores.min(), 0, scores.max()])
    ax.contour(xx, yy, scores, levels=[0], colors="black", linewidths=1)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, label="Настоящие")
    ax.scatter(X[y == -1, 0], X[y == -1, 1], s=22, label="Поддельные")

    support_original = scaler.inverse_transform(support_vectors)
    ax.scatter(
        support_original[:, 0],
        support_original[:, 1],
        s=75,
        facecolors="none",
        edgecolors="black",
        label="Опорные",
    )
    ax.set(title=title, xlabel="Верхнее поле, мм", ylabel="Длина, мм")
    ax.grid(alpha=0.2)


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, scaler, X, y = load_data()
    plot_dataset(X, y)

    print(f"Признаки: {', '.join(FEATURES)}")
    print(f"Train: {len(X_train)}, test: {len(X_test)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

    for row, kernel in enumerate(("linear", "rbf")):
        manual = fit_svm(X_train, y_train, kernel, C=C, gamma=GAMMA)
        reference = SVC(C=C, kernel=kernel, gamma=GAMMA, tol=1e-9)
        reference.fit(X_train, y_train)

        manual_prediction = predict(manual, X_test)
        reference_prediction = reference.predict(X_test)
        manual_scores = decision_function(manual, X_test)
        reference_scores = reference.decision_function(X_test)
        manual_metrics = calculate_metrics(y_test, manual_prediction)
        reference_metrics = calculate_metrics(y_test, reference_prediction)
        agreement = np.mean(manual_prediction == reference_prediction)
        score_difference = np.max(np.abs(manual_scores - reference_scores))

        print(f"\nЯдро: {kernel}")
        print(
            "Ручной SVM: "
            f"accuracy={manual_metrics['accuracy']:.4f}, "
            f"balanced_accuracy={manual_metrics['balanced_accuracy']:.4f}, "
            f"f1_fake={manual_metrics['f1_fake']:.4f}, "
            f"support={len(manual['alpha'])}"
        )
        print(
            "sklearn: "
            f"accuracy={reference_metrics['accuracy']:.4f}, "
            f"balanced_accuracy={reference_metrics['balanced_accuracy']:.4f}, "
            f"f1_fake={reference_metrics['f1_fake']:.4f}, "
            f"support={len(reference.support_)}"
        )
        print(f"Совпадение предсказаний: {agreement:.4f}")
        print(f"Максимальная разница f(x): {score_difference:.2e}")
        print(f"Ошибка ограничения: {manual['constraint_error']:.2e}")

        plot_boundary(
            axes[row, 0],
            lambda values, fitted=manual: decision_function(fitted, values),
            X,
            y,
            scaler,
            f"Ручной SVM, {kernel}",
            manual["support_vectors"],
        )
        plot_boundary(
            axes[row, 1],
            reference.decision_function,
            X,
            y,
            scaler,
            f"sklearn SVC, {kernel}",
            reference.support_vectors_,
        )

    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Границы классификации настоящих и поддельных банкнот")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "svm_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
