"""Эксперименты лабораторной №5: Ньютон–Рафсон, IRLS и sklearn."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
IMAGE_DIR = ROOT / "image"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from dataset import design_matrix, load_diabetes, to_signed
from logistic import (
    class_probability,
    fit_logistic,
    irls_direction,
    log_loss as sum_log_loss,
    newton_direction,
    predict_signed,
)

plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "font.size": 11})
TOL = 1e-6
STEP = 1.0


def save_fig(fig, name: str) -> None:
    IMAGE_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGE_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def self_check(F: np.ndarray, y: np.ndarray) -> float:
    """Один шаг из нуля и из случайной точки: направления Ньютона и IRLS совпадают."""
    gaps = []
    for w in (np.zeros(F.shape[1]), np.random.default_rng(0).normal(size=F.shape[1])):
        left = newton_direction(F, y, w)
        right = irls_direction(F, y, w)
        gaps.append(float(np.max(np.abs(left - right))))
    gap = max(gaps)
    if gap > 1e-8:
        raise RuntimeError(f"шаги Ньютона и IRLS разошлись на {gap}")
    return gap


def classification_row(name: str, y01: np.ndarray, proba: np.ndarray, signed: np.ndarray) -> dict:
    pred = np.where(signed > 0, 1.0, 0.0)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y01, pred)),
        "roc_auc": float(roc_auc_score(y01, proba)),
        "log_loss_mean": float(log_loss(y01, np.column_stack([1.0 - proba, proba]))),
    }


def plot_loss(newton, irls) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(newton.loss, color="#1f78b4", label="Ньютон–Рафсон")
    axes[0].plot(irls.loss, color="#ff7f00", linestyle="--", label="IRLS")
    axes[0].set_xlabel("итерация")
    axes[0].set_ylabel("Q(w) на обучении")
    axes[0].set_title("Убывание логарифмических потерь")
    axes[0].legend()

    axes[1].semilogy(np.arange(1, len(newton.steps) + 1), newton.steps, color="#1f78b4", label="Ньютон–Рафсон")
    axes[1].semilogy(
        np.arange(1, len(irls.steps) + 1),
        irls.steps,
        color="#ff7f00",
        linestyle="--",
        label="IRLS",
    )
    axes[1].axhline(TOL, color="#333333", linestyle=":", label=f"порог {TOL:g}")
    axes[1].set_xlabel("итерация")
    axes[1].set_ylabel("норма шага")
    axes[1].set_title("Длина шага")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "01_loss")


def plot_weights(names: list[str], w_newton: np.ndarray, w_irls: np.ndarray, w_ref: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    lo = min(w_newton.min(), w_irls.min(), w_ref.min())
    hi = max(w_newton.max(), w_irls.max(), w_ref.max())
    pad = 0.05 * (hi - lo + 1e-9)
    axes[0].scatter(w_ref, w_newton, s=28, c="#1f78b4", label="Ньютон–Рафсон")
    axes[0].scatter(w_ref, w_irls, s=28, facecolors="none", edgecolors="#ff7f00", label="IRLS")
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333", linewidth=1, label="совпадение")
    axes[0].set_xlabel("вес sklearn")
    axes[0].set_ylabel("наш вес")
    axes[0].set_title("Коэффициенты")
    axes[0].legend()

    order = np.argsort(np.abs(w_ref))
    ypos = np.arange(len(names))
    axes[1].barh(ypos - 0.15, w_newton[order], height=0.3, color="#1f78b4", label="Ньютон–Рафсон")
    axes[1].barh(ypos + 0.15, w_ref[order], height=0.3, color="#33a02c", label="sklearn")
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels([names[i] for i in order], fontsize=9)
    axes[1].set_xlabel("коэффициент")
    axes[1].set_title("Веса по признакам")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "02_weights")


def plot_roc(y_test: np.ndarray, curves: list[tuple[str, np.ndarray, str]]) -> None:
    fig, ax = plt.subplots()
    for name, proba, style in curves:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, style, linewidth=1.8, label=f"{name}, AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#333333", linestyle=":", label="случайный классификатор")
    ax.set_xlabel("доля ложных тревог")
    ax.set_ylabel("доля верных срабатываний")
    ax.set_title("ROC на тесте")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_roc")


def main() -> None:
    data = load_diabetes()
    F_train = design_matrix(data.X_train)
    F_test = design_matrix(data.X_test)
    y_train = to_signed(data.y_train)
    y_test = to_signed(data.y_test)
    step_gap = self_check(F_train, y_train)

    newton = fit_logistic(F_train, y_train, method="newton", step=STEP, tol=TOL)
    irls = fit_logistic(F_train, y_train, method="irls", step=STEP, tol=TOL)

    reference = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        fit_intercept=False,
        tol=TOL,
        max_iter=500,
    )
    reference.fit(F_train, data.y_train)
    w_ref = np.asarray(reference.coef_, dtype=np.float64).ravel()

    path_gap = float(np.max(np.abs(newton.path - irls.path)))
    weight_gap_methods = float(np.linalg.norm(newton.w - irls.w))
    weight_gap_ref = float(np.linalg.norm(newton.w - w_ref))
    proba_train = class_probability(F_train, newton.w)
    proba_test = class_probability(F_test, newton.w)
    proba_irls = class_probability(F_test, irls.w)
    proba_ref = reference.predict_proba(F_test)[:, 1]
    proba_gap = float(np.max(np.abs(proba_test - proba_ref)))

    names = list(data.feature_names) + ["свободный член"]
    rows = []
    for name, a, b, c in zip(names, newton.w, irls.w, w_ref):
        rows.append(
            {
                "name": name,
                "newton": float(a),
                "irls": float(b),
                "sklearn": float(c),
            }
        )

    metrics = [
        classification_row("newton", data.y_test, proba_test, predict_signed(F_test, newton.w)),
        classification_row("irls", data.y_test, proba_irls, predict_signed(F_test, irls.w)),
        classification_row("sklearn", data.y_test, proba_ref, np.where(reference.predict(F_test) >= 0.5, 1.0, -1.0)),
    ]

    # Квадратичность: отношение длины следующего шага к квадрату текущего, ближе к концу.
    ratios = []
    for prev, nxt in zip(newton.steps[:-1], newton.steps[1:]):
        ratios.append(float(nxt / prev**2) if prev > 0 else None)

    plot_loss(newton, irls)
    plot_weights(names, newton.w, irls.w, w_ref)
    plot_roc(
        data.y_test,
        [
            ("sklearn", proba_ref, "C2-"),
            ("Ньютон–Рафсон", proba_test, "C0--"),
            ("IRLS", proba_irls, "C1:"),
        ],
    )

    summary = {
        "zero_counts": data.zero_counts,
        "n_train": int(len(data.y_train)),
        "n_test": int(len(data.y_test)),
        "n_features": int(data.X_train.shape[1]),
        "train_positive": int(np.sum(data.y_train == 1)),
        "train_negative": int(np.sum(data.y_train == 0)),
        "test_positive": int(np.sum(data.y_test == 1)),
        "test_negative": int(np.sum(data.y_test == 0)),
        "tol": TOL,
        "step": STEP,
        "newton_iters": newton.n_iter,
        "irls_iters": irls.n_iter,
        "direction_gap": step_gap,
        "path_gap": path_gap,
        "weight_gap_methods": weight_gap_methods,
        "weight_gap_sklearn": weight_gap_ref,
        "proba_gap_test": proba_gap,
        "train_loss": newton.loss.tolist(),
        "steps": newton.steps.tolist(),
        "quadratic_ratios": ratios,
        "train_Q": sum_log_loss(F_train, y_train, newton.w),
        "sklearn_train_mean_log_loss": float(log_loss(data.y_train, reference.predict_proba(F_train))),
        "metrics": metrics,
        "weights": rows,
        "hessian_at_solution_log10_cond": float(
            np.log10(np.linalg.cond(F_train.T @ ((proba_train * (1 - proba_train))[:, None] * F_train)))
        ),
    }
    IMAGE_DIR.mkdir(exist_ok=True)
    (IMAGE_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "weights"}, ensure_ascii=False, indent=2))
    print("weights:")
    for row in rows:
        print(f"{row['name']}\t{row['newton']:.6f}\t{row['irls']:.6f}\t{row['sklearn']:.6f}")


if __name__ == "__main__":
    main()
