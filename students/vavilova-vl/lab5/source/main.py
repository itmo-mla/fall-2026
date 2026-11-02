"""Лабораторная работа №5: логистическая регрессия методом Ньютона-Рафсона и IRLS.

Запуск из каталога lab5:  uv run python source/main.py
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import plots
from data import add_bias, load_dataset
from losses import margins, sigmoid
from metrics import accuracy, confusion, mean_log_loss, roc_auc, roc_curve
from model import LogisticRegression
from reference import REFERENCE_SOLVERS, compare, fit_reference, irls_identity

IMAGES = Path(__file__).resolve().parent.parent / "images"
TOLERANCE = 1e-8


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_history(name: str, model: LogisticRegression) -> None:
    h = model.history_
    print(f"{name}: {h.n_iter} итераций")
    print(f"{'t':>3} {'Q(w_t)':>14} {'‖Q′‖∞':>10} {'h_t':>6} {'max|Δσ|':>10}")
    for t in range(len(h.loss)):
        step = f"{h.step[t - 1]:6.3f}" if t else "     —"
        ds = f"{h.sigma_change[t - 1]:10.2e}" if t else "         —"
        print(f"{t:>3} {h.loss[t]:14.8f} {h.grad_norm[t]:10.2e} {step} {ds}")


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    data = load_dataset()
    Xtr, ytr, Xte, yte = data.X_train, data.y_train, data.X_test, data.y_test

    section("Датасет")
    print(f"Titanic: обучение ℓ = {len(ytr)}, тест k = {len(yte)}, признаков n = {Xtr.shape[1]} + константа")
    print("Признаки:", ", ".join(data.feature_names))
    print(f"Доля выживших: обучение {ytr.mean():.3f}, тест {yte.mean():.3f}")

    section("Обучение")
    models = {
        "Ньютон-Рафсон, w⁰ = 0": LogisticRegression("newton", init="zero").fit(Xtr, ytr),
        "IRLS (слайд 9), w⁰ = МНК": LogisticRegression("irls", init="ols").fit(Xtr, ytr),
        "IRLS для GLM (слайд 19), w⁰ = МНК": LogisticRegression("irls_glm", init="ols").fit(Xtr, ytr),
    }
    for name, m in models.items():
        print_history(name, m)
        print()
    newton, irls, irls_glm = models.values()
    print("Попарные расхождения итоговых весов:")
    print(f"  Ньютон vs IRLS      : {np.max(np.abs(newton.weights - irls.weights)):.2e}")
    print(f"  IRLS vs IRLS-GLM    : {np.max(np.abs(irls.weights - irls_glm.weights)):.2e}")

    section("Эквивалентность шага IRLS и шага Ньютона")
    F, y_pm = add_bias(Xtr), 2 * ytr - 1
    w_ols = LogisticRegression().initial_weights(F, y_pm)
    # В w* градиент ≈ 1e−14, и относительная ошибка там бессмысленна — берём точки траектории.
    points = [("w = 0", np.zeros(F.shape[1])), ("w = МНК", w_ols)]
    points += [(f"w_{t} IRLS", irls.history_.w[t]) for t in (1, 2, 3)]
    for label, w in points:
        dh, dg = irls_identity(F, y_pm, w)
        print(f"{label:>10}:  |F̃ᵀF̃ − Q″| / |Q″| = {dh:.1e},   |F̃ᵀỹ + Q′| / |Q′| = {dg:.1e}")
    # Из одной точки все три метода должны пройти одну и ту же траекторию.
    same = {k: LogisticRegression(k, init="ols").fit(Xtr, ytr).history_ for k in ("newton", "irls", "irls_glm")}
    n_common = min(h.n_iter for h in same.values()) + 1
    for other in ("irls", "irls_glm"):
        dev = max(np.max(np.abs(a - b)) for a, b in zip(same["newton"].w[:n_common], same[other].w[:n_common]))
        print(f"Траектории из w = МНК, Ньютон vs {other:<8}: max_t |w_t − w_t′| = {dev:.2e} "
              f"({n_common - 1} итераций)")

    section("Сравнение с sklearn.linear_model.LogisticRegression(C=∞)")
    print(f"{'solver':>16} {'итер.':>6} {'max|Δw|':>10} {'max|Δp| тест':>13} {'разных классов':>15}")
    worst = 0.0
    for solver in REFERENCE_SOLVERS:
        c = compare(irls, Xtr, ytr, Xte, solver)
        print(f"{c.solver:>16} {c.n_iter:>6} {c.weights:10.2e} {c.proba:13.2e} {c.disagreements:>15}")
        if solver == "newton-cholesky":
            worst = max(c.weights, c.proba)
    print(f"Расхождение с newton-cholesky: {worst:.2e}  [{'OK' if worst < TOLERANCE else 'РАСХОЖДЕНИЕ'}, "
          f"допуск {TOLERANCE:g}]")

    section("Коэффициенты (IRLS)")
    ref = fit_reference(Xtr, ytr, "newton-cholesky")
    w_ref = np.concatenate([ref.intercept_, ref.coef_.ravel()])
    names = ["(константа)"] + data.feature_names
    print(f"{'признак':>12} {'w_j':>9} {'sklearn':>9} {'e^w_j':>7}")
    for name, w, wr in zip(names, irls.weights, w_ref):
        print(f"{name:>12} {w:9.4f} {wr:9.4f} {np.exp(w):7.3f}")

    section("Качество")
    p_ref_te = ref.predict_proba(Xte)[:, 1]
    print(f"{'':>10} {'accuracy':>9} {'log-loss':>9} {'AUC':>7}")
    for part, X, y in (("обучение", Xtr, ytr), ("тест", Xte, yte)):
        p = irls.predict_proba(X)
        print(f"{part:>10} {accuracy(y, irls.predict(X)):9.4f} {mean_log_loss(y, p):9.4f} {roc_auc(y, p):7.4f}")
    p_te = irls.predict_proba(Xte)
    print(f"sklearn, тест: accuracy {accuracy(yte, ref.predict(Xte)):.4f}, "
          f"log-loss {mean_log_loss(yte, p_ref_te):.4f}, AUC {roc_auc_score(yte, p_ref_te):.4f}")
    print(f"Своя AUC − roc_auc_score на тесте: {abs(roc_auc(yte, p_te) - roc_auc_score(yte, p_te)):.1e}")
    print("Матрица ошибок на тесте:", confusion(yte, irls.predict(Xte)))
    print(f"Q(w*) / ℓ = {irls.history_.loss[-1] / len(ytr):.6f} — средний log-loss на обучении")

    M = margins(F, y_pm, irls.weights)
    s = sigmoid(M)
    weights = (1 - s) * s
    print(f"Веса IRLS (1−σ)σ: max {weights.max():.4f}, ошибок на обучении {np.sum(M < 0)}, "
          f"средний вес ошибок {weights[M < 0].mean():.4f}, верных {weights[M >= 0].mean():.4f}")

    plots.convergence({name: m.history_.grad_norm for name, m in models.items()}, IMAGES / "convergence.png")
    plots.coefficients(names, irls.weights, w_ref, IMAGES / "coefficients.png")
    plots.irls_weights(M, weights, IMAGES / "irls_weights.png")
    fpr, tpr = roc_curve(yte, p_te)
    plots.roc(fpr, tpr, roc_auc(yte, p_te), IMAGES / "roc.png")
    print(f"\nГрафики сохранены в {IMAGES}")


if __name__ == "__main__":
    main()
