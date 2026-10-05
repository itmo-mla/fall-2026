import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_preparation import CLASS_NAMES, load_dataset
from svm import DualSVM, linear_kernel, polynomial_kernel, rbf_kernel
from visualization import (
    plot_decision_boundaries,
    plot_linear_weights,
    plot_margins,
    plot_score_comparison,
)

LAB_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = LAB_DIR / "graphs"
RESULTS_DIR = LAB_DIR / "results"

C = 1.0
DEGREE = 3
COEF0 = 1.0
REFERENCE_TOL = 1e-6


def kernel_configs(gamma: float) -> list[dict]:
    """Ядра своей реализации и те же ядра в параметрах sklearn."""
    return [
        {
            "name": "linear",
            "title": "Линейное ядро",
            "kernel": linear_kernel(),
            "sklearn": {"kernel": "linear"},
        },
        {
            "name": "poly",
            "title": f"Полиномиальное ядро (степень {DEGREE})",
            "kernel": polynomial_kernel(DEGREE, gamma, COEF0),
            "sklearn": {"kernel": "poly", "degree": DEGREE, "gamma": gamma, "coef0": COEF0},
        },
        {
            "name": "rbf",
            "title": "RBF-ядро",
            "kernel": rbf_kernel(gamma),
            "sklearn": {"kernel": "rbf", "gamma": gamma},
        },
    ]


def fit_pair(config: dict, X: np.ndarray, y: np.ndarray) -> tuple[DualSVM, SVC, float, float]:
    start = time.perf_counter()
    own = DualSVM(config["kernel"], C=C).fit(X, y)
    own_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference = SVC(C=C, tol=REFERENCE_TOL, **config["sklearn"]).fit(X, y)
    reference_seconds = time.perf_counter() - start
    return own, reference, own_seconds, reference_seconds


def reference_dual_objective(reference: SVC, kernel, X: np.ndarray) -> float:
    coef = reference.dual_coef_[0]
    support_vectors = X[reference.support_]
    return float(0.5 * coef @ kernel(support_vectors, support_vectors) @ coef - np.abs(coef).sum())


def quality(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def boundary_panels(X: np.ndarray, y: np.ndarray, gamma: float, with_accuracy: bool) -> list[dict]:
    panels = []
    for config in kernel_configs(gamma):
        own, reference, _, _ = fit_pair(config, X, y)
        title = config["title"]
        if with_accuracy:
            title += f", accuracy = {accuracy_score(y, own.predict(X)):.3f}"
        title += f"\nопорных: {len(own.support_)} (sklearn: {len(reference.support_)})"
        panels.append({"title": title, "own": own, "reference": reference})
        print(f"  {config['name']}: {own.n_iter_} итераций, KKT = {own.kkt_violation_:.1e}")
    return panels


def main() -> None:
    GRAPHS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    data = load_dataset()
    X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test
    gamma = 1.0 / X_train.shape[1]
    print(f"Train: {X_train.shape}, test: {X_test.shape}, gamma = {gamma:.4f}, C = {C}")

    rows = []
    score_panels = []
    linear_pair = None
    print("Обучение на 8 признаках")
    for config in kernel_configs(gamma):
        own, reference, own_seconds, reference_seconds = fit_pair(config, X_train, y_train)
        own_scores = own.decision_function(X_test)
        reference_scores = reference.decision_function(X_test)
        own_pred = own.predict(X_test)
        reference_pred = reference.predict(X_test)

        common = {
            "kernel": config["name"],
            "max_score_difference": np.abs(own_scores - reference_scores).max(),
            "prediction_agreement": np.mean(own_pred == reference_pred),
        }
        rows.append(
            {
                **common,
                "implementation": "Своя реализация",
                "train_accuracy": accuracy_score(y_train, own.predict(X_train)),
                **quality(y_test, own_pred),
                "n_support": len(own.support_),
                "w0": own.w0_,
                "dual_objective": own.dual_objective_,
                "kkt_violation": own.kkt_violation_,
                "fit_seconds": own_seconds,
            }
        )
        rows.append(
            {
                **common,
                "implementation": "Эталон: sklearn SVC",
                "train_accuracy": accuracy_score(y_train, reference.predict(X_train)),
                **quality(y_test, reference_pred),
                "n_support": len(reference.support_),
                "w0": -reference.intercept_[0],
                "dual_objective": reference_dual_objective(reference, config["kernel"], X_train),
                "kkt_violation": np.nan,
                "fit_seconds": reference_seconds,
            }
        )
        score_panels.append({"title": config["title"], "own": own_scores, "reference": reference_scores})
        if config["name"] == "linear":
            linear_pair = (own, reference)
        print(
            f"  {config['name']}: {own_seconds:.1f} c, {own.n_iter_} итераций ({own.message_}), "
            f"KKT = {own.kkt_violation_:.1e}, на границе зазора: {own.n_margin_}"
        )

    columns = [
        "kernel", "implementation", "train_accuracy", "accuracy", "precision", "recall", "f1",
        "n_support", "w0", "dual_objective", "kkt_violation", "fit_seconds",
        "max_score_difference", "prediction_agreement",
    ]
    metrics = pd.DataFrame(rows)[columns]
    metrics.to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False)
    print(metrics.drop(columns=["precision", "recall", "fit_seconds"]).to_string(index=False))

    # Линейный классификатор: явные веса w и порог w0, a(x) = sign(<w, x> - w0).
    own_linear, reference_linear = linear_pair
    own_weights = own_linear.linear_weights()
    reference_weights = reference_linear.coef_[0]
    weights = pd.DataFrame(
        {
            "feature": [*data.feature_names, "w0"],
            "w_own": [*own_weights, own_linear.w0_],
            "w_sklearn": [*reference_weights, -reference_linear.intercept_[0]],
        }
    )
    weights.to_csv(RESULTS_DIR / "linear_weights.csv", index=False)
    print(weights.to_string(index=False))

    train_margins = y_train * own_linear.decision_function(X_train)
    test_margins = y_test * own_linear.decision_function(X_test)
    print(
        "Отступы линейного SVM на train: "
        f"M < 1: {(train_margins < 1 - 1e-6).sum()}, "
        f"M = 1: {(np.abs(train_margins - 1) <= 1e-6).sum()}, "
        f"M > 1: {(train_margins > 1 + 1e-6).sum()}"
    )

    plot_linear_weights(data.feature_names, own_weights, reference_weights, GRAPHS_DIR / "linear_weights.png")
    plot_margins(
        [("Train", train_margins), ("Test", test_margins)],
        "Отступы линейного SVM",
        GRAPHS_DIR / "linear_margins.png",
    )
    plot_score_comparison(score_panels, y_test, CLASS_NAMES, GRAPHS_DIR / "decision_function_comparison.png")

    # Для рисунка модели обучаются заново на двух главных компонентах.
    print("Обучение на проекции PCA")
    pca = PCA(n_components=2).fit(X_train)
    explained = pca.explained_variance_ratio_
    print(f"  объяснённая дисперсия: {explained[0]:.3f}, {explained[1]:.3f}")
    X_train_2d = pca.transform(X_train)
    plot_decision_boundaries(
        boundary_panels(X_train_2d, y_train, gamma=0.5, with_accuracy=True),
        X_train_2d,
        y_train,
        CLASS_NAMES,
        "Разделяющие поверхности SVM на обучающей выборке Titanic (проекция PCA)",
        (f"Главная компонента 1 ({explained[0]:.0%} дисперсии)", f"Главная компонента 2 ({explained[1]:.0%} дисперсии)"),
        GRAPHS_DIR / "decision_boundaries_pca.png",
    )

    # Иллюстрация трюка с ядром на синтетических линейно неразделимых данных.
    print("Обучение на make_moons")
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_moons = StandardScaler().fit_transform(X_moons)
    y_moons = np.where(y_moons == 1, 1, -1)
    plot_decision_boundaries(
        boundary_panels(X_moons, y_moons, gamma=0.5, with_accuracy=True),
        X_moons,
        y_moons,
        {-1: "Класс", 1: "Класс"},
        "Трюк с ядром на линейно неразделимых данных (make_moons)",
        ("Признак 1", "Признак 2"),
        GRAPHS_DIR / "kernel_trick_moons.png",
    )


if __name__ == "__main__":
    main()
