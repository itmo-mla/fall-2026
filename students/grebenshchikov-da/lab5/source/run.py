"""Запуск эксперимента, проверок и построение графиков"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from data import load_and_prepare
from logistic_regression import (
    add_intercept,
    fit_irls,
    fit_newton,
    predict,
    predict_proba,
)


plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"
COLORS = {"newton": "#2F7775", "irls": "#9375A5", "reference": "#169B55"}


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Вычисляет метрики бинарной классификации"""
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
        "log_loss": log_loss(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def plot_convergence(newton_losses: np.ndarray, irls_losses: np.ndarray) -> None:
    """Строит график сходимости двух методов"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    newton_iterations = np.arange(1, len(newton_losses) + 1)
    irls_iterations = np.arange(1, len(irls_losses) + 1)

    ax.plot(
        newton_iterations,
        newton_losses,
        color=COLORS["newton"],
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        linewidth=2,
        label="Ньютон-Рафсон",
        zorder=3,
    )
    ax.plot(
        irls_iterations,
        irls_losses,
        color=COLORS["irls"],
        marker="s",
        markersize=4,
        linestyle="--",
        linewidth=1.8,
        label="IRLS",
        zorder=4,
    )
    ax.set_title("Сходимость методов")
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("Log-loss на обучающей выборке")
    ax.set_xticks(newton_iterations)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "convergence.png", dpi=180)
    plt.close(fig)


def plot_probability_comparison(
    manual_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
) -> None:
    """Сравнивает вероятности ручной и эталонной моделей"""
    max_difference = np.max(np.abs(manual_probabilities - reference_probabilities))
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(
        manual_probabilities,
        reference_probabilities,
        color=COLORS["newton"],
        alpha=0.65,
        s=34,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], linestyle="--", linewidth=2)
    ax.text(
        0.04,
        0.95,
        f"Максимальное расхождение: {max_difference:.2e}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    ax.set_title("Сравнение предсказанных вероятностей")
    ax.set_xlabel("Ручная реализация")
    ax.set_ylabel("sklearn")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "probability_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Выполняет полный эксперимент"""
    split = load_and_prepare()
    X_train = add_intercept(split.X_train)
    X_test = add_intercept(split.X_test)

    newton_result = fit_newton(X_train, split.y_train)
    irls_result = fit_irls(X_train, split.y_train)

    # sklearn решает ту же задачу без регуляризации
    reference = LogisticRegression(
        penalty=None,
        fit_intercept=False,
        solver="newton-cholesky",
        tol=1e-12,
        max_iter=1000,
    )
    reference.fit(X_train, split.y_train)

    newton_probabilities = predict_proba(X_test, newton_result.weights)
    irls_probabilities = predict_proba(X_test, irls_result.weights)
    reference_probabilities = reference.predict_proba(X_test)[:, 1]

    weight_difference = np.max(np.abs(newton_result.weights - irls_result.weights))
    reference_difference = np.max(
        np.abs(newton_probabilities - reference_probabilities)
    )
    reference_weight_difference = np.max(
        np.abs(newton_result.weights - reference.coef_[0])
    )

    assert split.X_train.shape == (675, 7)
    assert split.X_test.shape == (225, 7)
    assert np.allclose(split.X_train.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(split.X_train.std(axis=0), 1.0, atol=1e-12)
    assert newton_result.loss_history[-1] < newton_result.loss_history[0]
    assert irls_result.loss_history[-1] < irls_result.loss_history[0]
    assert newton_result.converged
    assert irls_result.converged
    assert weight_difference < 1e-10
    assert reference_weight_difference < 1e-8
    assert reference_difference < 1e-8
    assert np.array_equal(
        predict(X_test, newton_result.weights),
        predict(X_test, irls_result.weights),
    )
    assert np.array_equal(
        predict(X_test, newton_result.weights),
        reference.predict(X_test),
    )

    results = {
        "Ньютон-Рафсон": (
            calculate_metrics(split.y_test, newton_probabilities),
            newton_result.iterations,
        ),
        "IRLS": (
            calculate_metrics(split.y_test, irls_probabilities),
            irls_result.iterations,
        ),
        "sklearn": (
            calculate_metrics(split.y_test, reference_probabilities),
            int(reference.n_iter_[0]),
        ),
    }

    IMAGES_DIR.mkdir(exist_ok=True)
    plot_convergence(newton_result.loss_history, irls_result.loss_history)
    plot_probability_comparison(newton_probabilities, reference_probabilities)

    print("Метод              Точность  Сбаланс.  F1      Ошибка  ROC-AUC  Итерации")
    for method, (metrics, iterations) in results.items():
        print(
            f"{method:<18}"
            f"{metrics['accuracy']:.4f}    "
            f"{metrics['balanced_accuracy']:.4f}    "
            f"{metrics['f1']:.4f}  "
            f"{metrics['log_loss']:.4f}    "
            f"{metrics['roc_auc']:.4f}   "
            f"{iterations}"
        )

    print(f"Максимальная разница весов Ньютона и IRLS: {weight_difference:.2e}")
    print(f"Максимальная разница весов со sklearn: {reference_weight_difference:.2e}")
    print(f"Максимальная разница вероятностей со sklearn: {reference_difference:.2e}")


if __name__ == "__main__":
    main()
