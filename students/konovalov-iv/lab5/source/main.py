import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from dataset import add_bias, load_data
from logistic_regression import (LogisticRegression, gradient, irls_direction, log_loss,
                                 margin, newton_direction, sigmoid)
from metrics import classification_metrics, confusion_matrix
from plots import plot_convergence, plot_irls_weights, plot_separable, plot_weights

PLOTS = Path(__file__).resolve().parent / "plots"
SEED = 0


def separable_run(X, y, method, h=1.0, n_iter=30):
    w = np.zeros(X.shape[1])
    norms, Q = [0.0], [log_loss(w, X, y)]
    for _ in range(n_iter):
        try:
            step = newton_direction(w, X, y) if method == "newton" else h * gradient(w, X, y)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        w = w - step
        norms.append(np.linalg.norm(w))
        Q.append(log_loss(w, X, y))
    return norms, Q


def main():
    PLOTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"train: {X_train.shape}, test: {X_test.shape}, доля класса +1: {np.mean(y_train == 1):.2f}")

    lipschitz = np.linalg.eigvalsh(X_train.T @ X_train).max() / 4
    models = {
        "Ньютон-Рафсон": LogisticRegression(method="newton").fit(X_train, y_train),
        "IRLS": LogisticRegression(method="irls").fit(X_train, y_train),
        "градиентный спуск": LogisticRegression(method="gd", h=1 / lipschitz, max_iter=5000).fit(X_train, y_train),
    }

    reference = SklearnLogisticRegression(C=np.inf, solver="newton-cg", tol=1e-10, max_iter=1000)
    start = time.perf_counter()
    reference.fit(X_train[:, 1:], y_train)
    reference_time = time.perf_counter() - start
    w_reference = np.concatenate([reference.intercept_, reference.coef_[0]])

    w_check = np.linalg.lstsq(X_train, y_train.astype(float), rcond=None)[0]
    step_difference = np.max(np.abs(irls_direction(w_check, X_train, y_train)
                                    + newton_direction(w_check, X_train, y_train)))
    print(f"\nшаг IRLS + шаг Ньютона в точке МНК-решения: max|Δ| = {step_difference:.2e}")

    print("\n| Модель | Итераций | Q (train) | ‖Q'‖ | max\\|w - w_sklearn\\| | max\\|p - p_sklearn\\| (test) "
          "| Accuracy | Precision | Recall | F1 | Время, с |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    p_reference = reference.predict_proba(X_test[:, 1:])[:, 1]
    for name, model in models.items():
        r = classification_metrics(y_test, model.predict(X_test))
        print(f"| {name} | {len(model.history['Q']) - 1} | {model.history['Q'][-1]:.6f} "
              f"| {model.history['grad_norm'][-1]:.1e} | {np.max(np.abs(model.w - w_reference)):.1e} "
              f"| {np.max(np.abs(model.predict_proba(X_test) - p_reference)):.1e} "
              f"| {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} "
              f"| {model.history['time']:.4f} |")
    r = classification_metrics(y_test, reference.predict(X_test[:, 1:]))
    print(f"| sklearn LogisticRegression | {reference.n_iter_[0]} | {log_loss(w_reference, X_train, y_train):.6f} "
          f"| {np.linalg.norm(gradient(w_reference, X_train, y_train)):.1e} | 0 | 0 "
          f"| {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} "
          f"| {reference_time:.4f} |")

    newton = models["Ньютон-Рафсон"]
    same = np.mean(newton.predict(X_test) == reference.predict(X_test[:, 1:]))
    print(f"\nсовпадение предсказаний Ньютон-Рафсон и sklearn на test: {same:.3f}")
    print("матрица ошибок на test [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, newton.predict(X_test)))

    print("\n| Признак | w (Ньютон) | w (IRLS) | w (sklearn) | exp(w) |")
    print("|---|---|---|---|---|")
    for j, name in enumerate(feature_names):
        print(f"| {name} | {newton.w[j]:.4f} | {models['IRLS'].w[j]:.4f} | {w_reference[j]:.4f} "
              f"| {np.exp(newton.w[j]):.3f} |")

    Q_optimal = min(model.history["Q"][-1] for model in models.values())
    plot_convergence(models, Q_optimal, PLOTS / "convergence.png")
    plot_weights({"Ньютон-Рафсон": newton.w, "IRLS": models["IRLS"].w, "sklearn": w_reference},
                 feature_names, PLOTS / "weights.png")
    M = margin(models["IRLS"].w, X_train, y_train)
    sigma = sigmoid(M)
    plot_irls_weights(M, (1 - sigma) * sigma, PLOTS / "irls_weights.png")

    X_blobs, y_blobs = make_blobs(n_samples=200, centers=[(-2, -2), (2, 2)], cluster_std=1.0, random_state=SEED)
    X_blobs, y_blobs = add_bias(X_blobs), np.where(y_blobs == 1, 1, -1)
    print(f"\nразделимая выборка: отступы МНК-решения все > 0: "
          f"{np.all(margin(np.linalg.lstsq(X_blobs, y_blobs.astype(float), rcond=None)[0], X_blobs, y_blobs) > 0)}")
    lipschitz_blobs = np.linalg.eigvalsh(X_blobs.T @ X_blobs).max() / 4
    runs = {"Ньютон-Рафсон": separable_run(X_blobs, y_blobs, "newton", n_iter=200),
            "градиентный спуск": separable_run(X_blobs, y_blobs, "gd", h=1 / lipschitz_blobs, n_iter=200)}
    for name, (norms, Q) in runs.items():
        print(f"{name}: итераций {len(norms) - 1}, ‖w‖ = {norms[-1]:.1f}, Q = {Q[-1]:.2e}")
    plot_separable(runs, PLOTS / "separable.png")
    print(f"\nграфики сохранены в {PLOTS}")


if __name__ == "__main__":
    main()
