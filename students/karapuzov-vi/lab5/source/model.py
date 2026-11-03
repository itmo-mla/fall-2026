import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from data import PLOTS_DIR, as_numpy, load_raw_dataframe, prepare_data

MAX_ITER = 30
STEP_TOL = 1e-12
BOX_COLOR = "#4C72B0"
POSITIVE_COLOR = "#C44E52"
WEIGHT_LABELS = {
    "intercept": "свободный член",
    "CRIM": "преступность",
    "ZN": "крупные участки",
    "INDUS": "промзоны",
    "CHAS": "у реки Чарльз",
    "NOX": "оксиды азота",
    "RM": "комнаты",
    "AGE": "дома старше 1940",
    "DIS": "до центров занятости",
    "RAD": "доступ к шоссе",
    "TAX": "налог",
    "PTRATIO": "учеников на учителя",
    "LSTAT": "низкий статус",
}


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def mean_log_likelihood(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    probabilities = np.clip(sigmoid(X @ weights), 1e-15, 1.0 - 1e-15)
    return float(np.mean(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities)))


def _weighted_gram(X: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return X.T @ (X * variance[:, None])


def newton_raphson(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = MAX_ITER,
    step_tol: float = STEP_TOL,
) -> dict:
    weights = np.zeros(X.shape[1], dtype=float)
    history = [mean_log_likelihood(X, y, weights)]
    for _ in range(max_iter):
        probabilities = sigmoid(X @ weights)
        variance = probabilities * (1.0 - probabilities)
        gradient = X.T @ (y - probabilities)
        step = np.linalg.solve(_weighted_gram(X, variance), gradient)
        weights = weights + step
        history.append(mean_log_likelihood(X, y, weights))
        if float(np.max(np.abs(step))) < step_tol:
            break
    return {"weights": weights, "history": np.asarray(history), "n_iter": len(history) - 1}


def irls(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = MAX_ITER,
    step_tol: float = STEP_TOL,
) -> dict:
    weights = np.zeros(X.shape[1], dtype=float)
    history = [mean_log_likelihood(X, y, weights)]
    for _ in range(max_iter):
        eta = X @ weights
        probabilities = sigmoid(eta)
        variance = probabilities * (1.0 - probabilities)
        weighted_response = variance * eta + (y - probabilities)
        updated = np.linalg.solve(_weighted_gram(X, variance), X.T @ weighted_response)
        step = updated - weights
        weights = updated
        history.append(mean_log_likelihood(X, y, weights))
        if float(np.max(np.abs(step))) < step_tol:
            break
    return {"weights": weights, "history": np.asarray(history), "n_iter": len(history) - 1}


def probabilities(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return sigmoid(X @ weights)


def _reference_weights(X: np.ndarray, y: np.ndarray, solver: str) -> tuple[np.ndarray, int]:
    model = LogisticRegression(
        C=np.inf,
        solver=solver,
        fit_intercept=False,
        max_iter=500,
        tol=1e-12,
    )
    model.fit(X, y)
    return model.coef_.ravel().astype(float), int(model.n_iter_[0])


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    predicted = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "log_loss": float(log_loss(y, proba)),
    }


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def run_comparison(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    newton = newton_raphson(X_train, y_train)
    weighted = irls(X_train, y_train)
    cholesky, cholesky_iter = _reference_weights(X_train, y_train, "newton-cholesky")
    lbfgs, lbfgs_iter = _reference_weights(X_train, y_train, "lbfgs")

    print("\n=== Сходимость ===")
    print(f"Ньютон–Рафсон, шагов: {newton['n_iter']}")
    print(f"IRLS, шагов: {weighted['n_iter']}")
    print(f"sklearn newton-cholesky, шагов: {cholesky_iter}")
    print(f"sklearn lbfgs, шагов: {lbfgs_iter}")
    print(
        "среднее правдоподобие train, старт / Ньютон / IRLS:",
        round(float(newton["history"][0]), 6),
        round(float(newton["history"][-1]), 6),
        round(float(weighted["history"][-1]), 6),
    )
    same_length = len(newton["history"]) == len(weighted["history"])
    history_gap = (
        _max_abs(newton["history"], weighted["history"]) if same_length else float("nan")
    )
    print("макс. |правдоподобие Ньютона − IRLS| по шагам:", history_gap)

    print("\n=== Расхождение весов, max |Δ| ===")
    gaps = {
        "newton_irls": _max_abs(newton["weights"], weighted["weights"]),
        "newton_cholesky": _max_abs(newton["weights"], cholesky),
        "newton_lbfgs": _max_abs(newton["weights"], lbfgs),
        "irls_cholesky": _max_abs(weighted["weights"], cholesky),
    }
    for name, gap in gaps.items():
        print(f"  {name}: {gap:.3e}")

    newton_train = probabilities(X_train, newton["weights"])
    newton_test = probabilities(X_test, newton["weights"])
    irls_test = probabilities(X_test, weighted["weights"])
    cholesky_test = probabilities(X_test, cholesky)
    lbfgs_test = probabilities(X_test, lbfgs)
    print("\n=== Расхождение вероятностей на test, max |Δ| ===")
    print("  Ньютон и IRLS:", f"{_max_abs(newton_test, irls_test):.3e}")
    print("  Ньютон и newton-cholesky:", f"{_max_abs(newton_test, cholesky_test):.3e}")
    print("  Ньютон и lbfgs:", f"{_max_abs(newton_test, lbfgs_test):.3e}")

    print("\n=== Качество, порог вероятности 0.5 ===")
    rows = {
        "Ньютон, train": _metrics(y_train, newton_train),
        "Ньютон, test": _metrics(y_test, newton_test),
        "IRLS, test": _metrics(y_test, irls_test),
        "newton-cholesky, test": _metrics(y_test, cholesky_test),
        "lbfgs, test": _metrics(y_test, lbfgs_test),
    }
    print(f"{'модель':<28} {'доля верных':>12} {'F1':>8} {'ROC-AUC':>8} {'log-loss':>10}")
    for name, row in rows.items():
        print(
            f"{name:<28} {row['accuracy']:12.4f} {row['f1']:8.4f} "
            f"{row['roc_auc']:8.4f} {row['log_loss']:10.4f}"
        )

    print("\n=== Веса Ньютона ===")
    return {
        "newton": newton,
        "irls": weighted,
        "cholesky": cholesky,
        "lbfgs": lbfgs,
        "gaps": gaps,
        "metrics": rows,
    }


def _plot_likelihood(newton_history: np.ndarray, irls_history: np.ndarray) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    steps = np.arange(len(newton_history))
    ax.plot(steps, newton_history, color=BOX_COLOR, linewidth=2, label="Ньютон–Рафсон")
    ax.scatter(np.arange(len(irls_history)), irls_history, color=POSITIVE_COLOR, s=28, zorder=3, label="IRLS")
    ax.set_title("Среднее правдоподобие на train")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Среднее логарифмическое правдоподобие")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_loglik.png", dpi=120)
    plt.close(fig)


def _plot_weights(feature_names: list[str], newton: np.ndarray, reference: np.ndarray) -> None:
    labels = [WEIGHT_LABELS.get(name, name) for name in feature_names]
    order = np.argsort(newton)
    labels = [labels[index] for index in order]
    newton = newton[order]
    reference = reference[order]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(positions - 0.18, newton, height=0.36, color=BOX_COLOR, label="Ньютон–Рафсон")
    ax.barh(positions + 0.18, reference, height=0.36, color=POSITIVE_COLOR, label="sklearn newton-cholesky")
    ax.set_yticks(positions, labels)
    ax.axvline(0.0, color="#333333", linewidth=0.8, label="нулевой вес")
    ax.set_title("Веса логистической регрессии")
    ax.set_xlabel("Вес признака после стандартизации")
    ax.set_ylabel("Признак")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_weights.png", dpi=120)
    plt.close(fig)


def print_weights(feature_names: list[str], weights: np.ndarray) -> None:
    for name, value in zip(feature_names, weights):
        print(f"  {name:<12} {value: .6f}")


if __name__ == "__main__":
    prepared = prepare_data(load_raw_dataframe())
    X_train, X_test, y_train, y_test = as_numpy(prepared)
    result = run_comparison(X_train, X_test, y_train, y_test)
    print_weights(prepared["feature_names"], result["newton"]["weights"])
    _plot_likelihood(result["newton"]["history"], result["irls"]["history"])
    _plot_weights(prepared["feature_names"], result["newton"]["weights"], result["cholesky"])
    print(f"\nГрафики: {PLOTS_DIR}")
