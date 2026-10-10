"""Эксперименты лабораторной №3: двойственный SVM, ядра, сравнение со sklearn."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
IMAGE_DIR = ROOT / "image"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from dataset import social_ads, social_ads_with_noise
from svm import SV_TOL, SVMModel, fit_svm, kernel_matrix

plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "font.size": 11,
    }
)

COLOR = {-1.0: "#1f78b4", 1.0: "#ff7f00"}
KERNELS = (
    ("linear", "линейное"),
    ("poly", "полиномиальное, d = 3"),
    ("rbf", "гауссово"),
)
C_GRID = (0.05, 0.2, 1.0, 5.0, 50.0)


def save_fig(fig, name: str) -> Path:
    IMAGE_DIR.mkdir(exist_ok=True)
    path = IMAGE_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def quality(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "errors": int(np.sum(y_true != y_pred)),
        "n": int(len(y_true)),
    }


def fit_timed(X, y, **kwargs) -> tuple[SVMModel, float]:
    start = time.perf_counter()
    model = fit_svm(X, y, **kwargs)
    return model, time.perf_counter() - start


def sklearn_fit(X, y, kernel: str, C: float, degree: int = 3, gamma: float = 0.5) -> SVC:
    params: dict = {"C": C, "kernel": kernel, "tol": 1e-5}
    if kernel == "poly":
        params.update(degree=degree, gamma=1.0, coef0=1.0)
    elif kernel == "rbf":
        params.update(gamma=gamma)
    model = SVC(**params)
    model.fit(X, y)
    return model


def dual_alignment(ours: SVMModel, sk: SVC) -> dict[str, float]:
    """λ_i y_i против dual_coef_, w_0 против −intercept_."""
    coef = np.zeros(len(ours.lam), dtype=np.float64)
    coef[sk.support_] = sk.dual_coef_.ravel()
    ours_coef = ours.lam * ours.y
    ours_sv = set(ours.support.tolist())
    sk_sv = set(int(i) for i in sk.support_)
    shared = np.array(sorted(ours_sv & sk_sv), dtype=int)
    gap = ours_coef - coef
    return {
        "n_support_ours": int(len(ours.support)),
        "n_support_sklearn": int(len(sk.support_)),
        "n_shared": int(len(shared)),
        "n_only_ours": int(len(ours_sv - sk_sv)),
        "n_only_sklearn": int(len(sk_sv - ours_sv)),
        "max_abs_dual": float(np.max(np.abs(gap))) if len(gap) else 0.0,
        "max_abs_dual_shared": float(np.max(np.abs(gap[shared]))) if len(shared) else 0.0,
        "w0": float(ours.w0),
        "sklearn_intercept": float(sk.intercept_[0]),
        "bias_gap": float(ours.w0 + sk.intercept_[0]),
    }


def scatter_classes(ax, X: np.ndarray, y: np.ndarray, support: np.ndarray | None = None) -> None:
    for label, title in ((-1.0, "не купил, −1"), (1.0, "купил, +1")):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=COLOR[label],
            s=28,
            label=title,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    if support is not None and len(support):
        ax.scatter(
            X[support, 0],
            X[support, 1],
            s=130,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            label="опорные векторы",
            zorder=4,
        )


def draw_margin(ax, model: SVMModel, X: np.ndarray) -> None:
    pad = 0.55
    x1 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 280)
    x2 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 280)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    if model.X.shape[1] != 2:
        raise ValueError("полоса рисуется только в двух признаках")
    zz = model.scores(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-80, 0, 80], colors=("#d0e2f2", "#fde0c2"), alpha=0.85, zorder=0)
    ax.contour(
        xx,
        yy,
        zz,
        levels=[-1, 0, 1],
        colors=("0.25", "black", "0.25"),
        linestyles=("--", "-", "--"),
        linewidths=(1.0, 1.6, 1.0),
    )
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])


def margin_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="black", linewidth=1.6, label="граница, отступ 0"),
            Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.0, label="границы полосы, отступ ±1"),
        ]
    )
    labels.extend(["граница, отступ 0", "границы полосы, отступ ±1"])
    ax.legend(handles, labels, loc="best", fontsize=8)


def plot_dataset(split, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter_classes(ax, split.X_train, split.y_train)
    ax.set_xlabel(split.feature_names[0] + ", после нормировки")
    ax.set_ylabel(split.feature_names[1] + ", после нормировки")
    ax.set_title(title)
    ax.legend(loc="best")
    save_fig(fig, name)


def plot_one_margin(model: SVMModel, X, y, names, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    draw_margin(ax, model, X)
    scatter_classes(ax, X, y, model.support)
    ax.set_xlabel(names[0] + ", после нормировки")
    ax.set_ylabel(names[1] + ", после нормировки")
    ax.set_title(title)
    margin_legend(ax)
    save_fig(fig, name)


def plot_c_panels(split) -> list[dict]:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    rows = []
    shown = (0.05, 1.0, 50.0)
    for ax, C in zip(axes, shown):
        model, seconds = fit_timed(split.X_train, split.y_train, kernel="linear", C=C)
        draw_margin(ax, model, split.X_train)
        scatter_classes(ax, split.X_train, split.y_train, model.support)
        test_q = quality(split.y_test, model.predict(split.X_test))
        ax.set_title(f"C = {C:g}, опорных {len(model.support)}")
        ax.set_xlabel(split.feature_names[0])
        rows.append(pack(model, split, seconds, test_q))
    axes[0].set_ylabel(split.feature_names[1])
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.6))
    labels.append("граница")
    handles.append(Line2D([0], [0], color="0.25", linestyle="--"))
    labels.append("полоса ±1")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_fig(fig, "03_regularization")
    return rows


def plot_weights(model: SVMModel, names: tuple[str, ...]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    magnitude = np.abs(model.w)
    colors = ("#1f78b4", "#ff7f00", "0.55")
    ax.bar(names, magnitude, color=colors, label="модуль веса")
    ax.set_ylabel("|w|")
    ax.set_title("Веса линейного SVM: два полезных признака и шум")
    ax.legend(loc="upper right")
    for i, value in enumerate(magnitude):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(magnitude) * 1.25)
    save_fig(fig, "04_weights")


def plot_kernels(split) -> list[dict]:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    rows = []
    for ax, (kernel, title) in zip(axes, KERNELS):
        model, seconds = fit_timed(split.X_train, split.y_train, kernel=kernel, C=1.0, degree=3, gamma=0.5)
        draw_margin(ax, model, split.X_train)
        scatter_classes(ax, split.X_train, split.y_train, model.support)
        test_q = quality(split.y_test, model.predict(split.X_test))
        ax.set_title(f"{title}\nошибок на тесте {test_q['errors']}/{test_q['n']}")
        ax.set_xlabel(split.feature_names[0])
        rows.append(pack(model, split, seconds, test_q, kernel=kernel))
    axes[0].set_ylabel(split.feature_names[1])
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.6))
    labels.append("граница")
    handles.append(Line2D([0], [0], color="0.25", linestyle="--"))
    labels.append("полоса ±1")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_fig(fig, "05_kernels")
    return rows


def plot_sklearn_pair(ours: SVMModel, sk: SVC, split) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    pad = 0.55
    X = split.X_train
    x1 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 280)
    x2 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 280)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    fields = (
        (ours.scores(grid), ours.support, "свой SVM, гауссово ядро"),
        (sk.decision_function(grid), sk.support_, "sklearn SVC, те же C и γ"),
    )
    for ax, (values, support, title) in zip(axes, fields):
        zz = values.reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-80, 0, 80], colors=("#d0e2f2", "#fde0c2"), alpha=0.85, zorder=0)
        ax.contour(
            xx,
            yy,
            zz,
            levels=[-1, 0, 1],
            colors=("0.25", "black", "0.25"),
            linestyles=("--", "-", "--"),
            linewidths=(1.0, 1.6, 1.0),
        )
        ax.set_xlim(x1[0], x1[-1])
        ax.set_ylim(x2[0], x2[-1])
        scatter_classes(ax, X, split.y_train, np.asarray(support, dtype=int))
        ax.set_title(title)
        ax.set_xlabel(split.feature_names[0])
    axes[0].set_ylabel(split.feature_names[1])
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.6))
    labels.append("граница")
    handles.append(Line2D([0], [0], color="0.25", linestyle="--"))
    labels.append("полоса ±1")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_fig(fig, "06_sklearn")


def plot_dual(ours: SVMModel, sk: SVC) -> None:
    coef = np.zeros(len(ours.lam))
    coef[sk.support_] = sk.dual_coef_.ravel()
    ours_coef = ours.lam * ours.y
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(coef, ours_coef, s=22, c="#1f78b4", label="объекты обучения")
    limit = float(max(np.max(np.abs(coef)), np.max(np.abs(ours_coef)), 1e-3)) * 1.15
    ax.plot([-limit, limit], [-limit, limit], color="black", linewidth=1, label="совпадение")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("sklearn, dual_coef_ = λ y")
    ax.set_ylabel("свой SVM, λ y")
    ax.set_title("Двойственные коэффициенты, гауссово ядро")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    save_fig(fig, "07_dual")


def pack(model: SVMModel, split, seconds: float, test_q: dict, kernel: str | None = None) -> dict:
    train_q = quality(split.y_train, model.predict(split.X_train))
    row = {
        "kernel": kernel or model.kernel,
        "C": model.C,
        "degree": model.degree,
        "gamma": model.gamma,
        "solver": model.solver,
        "seconds": seconds,
        "objective": model.objective,
        "constraint": model.constraint,
        "w0": model.w0,
        "n_support": int(len(model.support)),
        "n_boundary": int(len(model.boundary)),
        "n_train": int(len(split.y_train)),
        "train": train_q,
        "test": test_q,
    }
    if model.w is not None:
        row["w"] = model.w.tolist()
    return row


def sweep(split, kernel: str) -> list[dict]:
    rows = []
    for C in C_GRID:
        model, seconds = fit_timed(split.X_train, split.y_train, kernel=kernel, C=C, degree=3, gamma=0.5)
        test_q = quality(split.y_test, model.predict(split.X_test))
        rows.append(pack(model, split, seconds, test_q, kernel=kernel))
    return rows


def compare_kernel(split, kernel: str, C: float = 1.0) -> dict:
    ours, seconds = fit_timed(split.X_train, split.y_train, kernel=kernel, C=C, degree=3, gamma=0.5)
    sk = sklearn_fit(split.X_train, split.y_train, kernel, C=C, degree=3, gamma=0.5)
    ours_test = quality(split.y_test, ours.predict(split.X_test))
    sk_test = quality(split.y_test, sk.predict(split.X_test))
    mismatch = int(np.sum(ours.predict(split.X_test) != sk.predict(split.X_test)))
    row = pack(ours, split, seconds, ours_test, kernel=kernel)
    row["sklearn_test"] = sk_test
    row["test_mismatches"] = mismatch
    row["alignment"] = dual_alignment(ours, sk)
    if ours.w is not None:
        row["sklearn_coef"] = sk.coef_.ravel().tolist()
        row["coef_gap"] = float(np.max(np.abs(ours.w - sk.coef_.ravel())))
    return {"model": ours, "sklearn": sk, "row": row}


def self_check() -> None:
    same = kernel_matrix(np.zeros((1, 2)), np.zeros((1, 2)), "rbf")
    if abs(float(same.ravel()[0]) - 1.0) > 1e-12:
        raise RuntimeError("гауссово ядро в нуле должно быть 1")
    step = kernel_matrix(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), "linear")
    if abs(float(step.ravel()[0])) > 1e-12:
        raise RuntimeError("линейное ядро ортогональных векторов должно быть 0")
    poly = kernel_matrix(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), "poly", degree=2)
    if abs(float(poly.ravel()[0]) - 4.0) > 1e-12:
        raise RuntimeError("полиномиальное ядро (1+1)^2 должно быть 4")

    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(loc=(-2.2, -2.0), scale=0.25, size=(25, 2)),
            rng.normal(loc=(2.2, 2.0), scale=0.25, size=(25, 2)),
        ]
    )
    y = np.array([-1.0] * 25 + [1.0] * 25)
    model = fit_svm(X, y, kernel="linear", C=10.0)
    if np.any(model.predict(X) != y):
        raise RuntimeError("линейный SVM не разделил далёкие сгустки")
    sk = sklearn_fit(X, y, "linear", C=10.0)
    if np.any(model.predict(X) != sk.predict(X)):
        raise RuntimeError("на разделимой выборке предсказания разошлись со sklearn")
    gap = np.max(np.abs(model.scores(X) - sk.decision_function(X)))
    if gap > 0.15:
        raise RuntimeError(f"решающая функция далеко от sklearn, максимум {gap}")
    if len(model.boundary):
        margin = model.y[model.boundary] * model.scores(model.X[model.boundary])
        if np.max(np.abs(margin - 1.0)) > 0.05:
            raise RuntimeError("на граничных опорных отступ не равен 1")


def main() -> None:
    self_check()
    ads = social_ads()
    noisy = social_ads_with_noise()

    plot_dataset(ads, "01_dataset", "Social Network Ads, обучающая часть")
    linear, linear_seconds = fit_timed(ads.X_train, ads.y_train, kernel="linear", C=1.0)
    linear_test = quality(ads.y_test, linear.predict(ads.X_test))
    plot_one_margin(
        linear,
        ads.X_train,
        ads.y_train,
        ads.feature_names,
        f"Линейный SVM, C = 1, опорных {len(linear.support)}",
        "02_margin",
    )
    c_shown = plot_c_panels(ads)
    weights_model, weights_seconds = fit_timed(noisy.X_train, noisy.y_train, kernel="linear", C=1.0)
    plot_weights(weights_model, noisy.feature_names)
    kernel_rows = plot_kernels(ads)

    comparisons = {}
    sk_models = {}
    for kernel, _title in KERNELS:
        found = compare_kernel(ads, kernel, C=1.0)
        comparisons[kernel] = found["row"]
        sk_models[kernel] = found
    plot_sklearn_pair(sk_models["rbf"]["model"], sk_models["rbf"]["sklearn"], ads)
    plot_dual(sk_models["rbf"]["model"], sk_models["rbf"]["sklearn"])

    linear_cmp = compare_kernel(ads, "linear", C=1.0)
    linear_sweep = sweep(ads, "linear")
    rbf_sweep = sweep(ads, "rbf")

    summary = {
        "n_total": int(ads.n_neg + ads.n_pos),
        "n_neg": int(ads.n_neg),
        "n_pos": int(ads.n_pos),
        "n_train": int(len(ads.y_train)),
        "n_test": int(len(ads.y_test)),
        "linear_c1": pack(linear, ads, linear_seconds, linear_test, kernel="linear"),
        "c_panels": c_shown,
        "linear_sweep": linear_sweep,
        "weights": pack(
            weights_model,
            noisy,
            weights_seconds,
            quality(noisy.y_test, weights_model.predict(noisy.X_test)),
            kernel="linear",
        ),
        "kernel_rows": kernel_rows,
        "comparisons": comparisons,
        "linear_comparison": linear_cmp["row"],
        "rbf_sweep": rbf_sweep,
        "sv_tol": SV_TOL,
    }
    IMAGE_DIR.mkdir(exist_ok=True)
    (IMAGE_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
