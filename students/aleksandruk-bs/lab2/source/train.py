"""Эксперименты лабораторной №2: LOO, сравнение со sklearn, отбор эталонов."""

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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
IMAGE_DIR = ROOT / "image"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from dataset import load_wine_data, standardize, stratified_split
from knn import (
    gaussian_kernel,
    loo_curve,
    loo_distance_matrix,
    loo_predictions,
    pairwise_minkowski,
    predict,
    predict_loo_parzen_subset,
)
from prototypes import greedy_condense, lecture_roles, predict_1nn, predict_loo_1nn

plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "font.size": 11,
    }
)

CLASS_COLORS = ("#1f78b4", "#ff7f00", "#33a02c")
REGION_CMAP = ListedColormap(("#c6dbef", "#fdd0a2", "#c7e9c0"))
ROLE_MARKERS = {"эталон": "o", "неинформативный": ".", "шум": "X"}
ROLE_SIZES = {"эталон": 78, "неинформативный": 16, "шум": 70}


def save_fig(fig, name: str) -> Path:
    IMAGE_DIR.mkdir(exist_ok=True)
    path = IMAGE_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def quality(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "error": float(np.mean(y_true != y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def self_check() -> None:
    """Короткий контроль формул на двух хорошо разделимых сгустках."""
    zero = np.array(0.0)
    one = np.array(1.0)
    if abs(float(gaussian_kernel(zero)) - 1.0) > 1e-12:
        raise RuntimeError("K(0) должно быть 1")
    if abs(float(gaussian_kernel(one)) - float(np.exp(-2.0))) > 1e-12:
        raise RuntimeError("K(1) должно быть e^{-2}")

    toy = np.array([[0.0, 0.0], [1.0, 3.0]], dtype=np.float64)
    plain = pairwise_minkowski(toy, toy)
    weighted = pairwise_minkowski(toy, toy, feature_weights=np.array([1.0, 0.0]))
    if not weighted[0, 1] < plain[0, 1]:
        raise RuntimeError("вес признака не изменил расстояние Минковского")

    rng = np.random.default_rng(0)
    left = rng.normal(loc=-3.0, scale=0.25, size=(25, 2))
    right = rng.normal(loc=3.0, scale=0.25, size=(25, 2))
    X = np.vstack([left, right])
    y = np.array([0] * 25 + [1] * 25)
    dist = loo_distance_matrix(X)
    pred = loo_predictions(dist, y, k=5, mode="parzen")
    if float(np.mean(pred == y)) < 0.95:
        raise RuntimeError("Парзен не разделил контрольные сгустки")
    pred_nn = predict_loo_1nn(dist, y)
    pred_uniform = loo_predictions(dist, y, k=1, mode="uniform")
    if not np.array_equal(pred_nn, pred_uniform):
        raise RuntimeError("1NN и равномерное голосование при k=1 разошлись")
    selected = greedy_condense(dist, y)
    if selected.history_loo[-1] > selected.history_loo[0] + 1e-12:
        raise RuntimeError("жадное удаление увеличило Q")
    if len(np.unique(y[selected.mask])) < 2:
        raise RuntimeError("из эталонов пропал класс")


def sklearn_loo(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Эталон: равномерный KNN из библиотеки, каждый объект по очереди вынут."""
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform",
        metric="euclidean",
        algorithm="brute",
    )
    return cross_val_predict(model, X, y, cv=LeaveOneOut())


def time_call(fn, repeats: int) -> tuple[float, object]:
    last = None
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), last


def plot_dataset(X: np.ndarray, y: np.ndarray, names: list[str], class_names: list[str]) -> None:
    X_std, _ = standardize(X)
    std_raw = X.std(axis=0)
    std_scaled = X_std.std(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    counts = np.bincount(y)
    axes[0].bar(class_names, counts, color=CLASS_COLORS)
    axes[0].set_ylabel("Число объектов")
    axes[0].set_title("Баланс классов")

    axes[1].bar(range(len(names)), std_raw, color="0.45", label="СКО до нормировки")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("СКО (лог. шкала)")
    axes[1].set_title("Масштаб признаков до стандартизации")
    axes[1].legend(loc="upper left")

    axes[2].bar(range(len(names)), std_scaled, color="tab:blue", label="СКО после стандартизации")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1, label="единичный разброс")
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    axes[2].set_ylim(0.0, 1.45)
    axes[2].set_ylabel("СКО")
    axes[2].set_title("После приведения к одному масштабу")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "01_dataset")


def plot_margins(margins: np.ndarray, roles: np.ndarray) -> None:
    order = np.argsort(margins)
    xs = np.arange(1, len(margins) + 1)
    fig, ax = plt.subplots()
    shown = set()
    for role, color in (("шум", "tab:red"), ("неинформативный", "0.55"), ("эталон", "tab:blue")):
        mask = roles[order] == role
        if not np.any(mask):
            continue
        ax.plot(
            xs[mask],
            margins[order][mask],
            "o",
            markersize=4,
            color=color,
            label=role,
        )
        shown.add(role)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="M = 0")
    ax.set_xlabel("Объекты по возрастанию отступа")
    ax.set_ylabel("Отступ M")
    ax.set_title("Отступ 1NN и роль объекта после отбора эталонов")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "02_margins")
    return shown


def _leading_worse(ks: np.ndarray, risks: np.ndarray, k_star: int, tol: float) -> int:
    """Сколько первых k подряд хуже минимума больше чем на tol. Верхняя граница не включается."""
    rmin = float(risks[np.where(ks == k_star)[0][0]])
    last = 0
    for k, risk in zip(ks, risks):
        if int(k) >= k_star or float(risk) <= rmin + tol:
            break
        last = int(k)
    return last


def _draw_loo(ax, ks_parzen, risks_parzen, ks_uniform, risks_uniform, k_max: int | None) -> None:
    if k_max is None:
        p_slice = slice(None)
        u_slice = slice(None)
    else:
        p_slice = ks_parzen <= k_max
        u_slice = ks_uniform <= k_max
    kp, rp = ks_parzen[p_slice], risks_parzen[p_slice]
    ku, ru = ks_uniform[u_slice], risks_uniform[u_slice]
    k_parzen = int(ks_parzen[int(np.argmin(risks_parzen))])
    k_uniform = int(ks_uniform[int(np.argmin(risks_uniform))])
    ax.plot(kp, rp, color="tab:blue", linewidth=1.8, label="окно Парзена, гауссово ядро")
    ax.plot(ku, ru, color="tab:orange", linewidth=1.4, label="равномерный KNN (как в sklearn)")
    if k_parzen <= int(kp[-1]):
        ax.plot(
            k_parzen,
            float(risks_parzen[ks_parzen == k_parzen][0]),
            "o",
            color="tab:green",
            markersize=8,
            label=f"минимум Парзена, k={k_parzen}",
        )
    if k_uniform <= int(ku[-1]):
        ax.plot(
            k_uniform,
            float(risks_uniform[ks_uniform == k_uniform][0]),
            "s",
            color="tab:red",
            markersize=7,
            label=f"минимум равномерного KNN, k={k_uniform}",
        )
    ax.set_xlabel("Параметр k")
    ax.set_ylabel("Эмпирический риск LOO(k)")


def plot_loo(ks_parzen, risks_parzen, ks_uniform, risks_uniform, majority_error: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    k_uniform = int(ks_uniform[int(np.argmin(risks_uniform))])
    left_end = _leading_worse(ks_uniform, risks_uniform, k_uniform, tol=0.01)
    _draw_loo(axes[0], ks_parzen, risks_parzen, ks_uniform, risks_uniform, k_max=80)
    if left_end >= 1:
        axes[0].axvspan(1, left_end, color="tab:red", alpha=0.13, label="малое k: переобучение", zorder=0)
    axes[0].set_xlim(1, 80)
    axes[0].set_ylim(0.0, 0.08)
    axes[0].set_title("Небольшие k, масштаб у минимума")
    axes[0].legend(fontsize=8)

    _draw_loo(axes[1], ks_parzen, risks_parzen, ks_uniform, risks_uniform, k_max=None)
    rise = np.where(risks_uniform > 0.1)[0]
    if len(rise):
        axes[1].axvspan(
            int(ks_uniform[rise[0]]),
            int(ks_uniform[-1]),
            color="tab:purple",
            alpha=0.13,
            label="большое k: пересглаживание",
            zorder=0,
        )
    axes[1].axhline(
        majority_error,
        color="black",
        linestyle="--",
        linewidth=1,
        label="константа большинства",
    )
    axes[1].set_title("Вся сетка k")
    axes[1].legend(fontsize=8)
    fig.suptitle("Скользящий контроль: риск в зависимости от k", y=1.02)
    fig.tight_layout()
    save_fig(fig, "03_loo")


def plot_kernel() -> None:
    r = np.linspace(0.0, 2.5, 400)
    fig, ax = plt.subplots()
    ax.plot(r, gaussian_kernel(r), color="tab:blue", linewidth=2, label=r"$K(r)=\exp(-2r^{2})$")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="граница окна, r = 1")
    ax.scatter([0.0, 1.0], [1.0, float(np.exp(-2.0))], color="tab:red", zorder=3, label="K(0)=1, K(1)=e$^{-2}$")
    ax.set_xlabel("r = ρ(x, x_i) / h(x)")
    ax.set_ylabel("Вес K(r)")
    ax.set_title("Гауссово ядро окна Парзена")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "04_kernel")


def _regions(ax, grid_x, grid_y, pred, xx_shape) -> None:
    ax.contourf(
        grid_x,
        grid_y,
        pred.reshape(xx_shape),
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=REGION_CMAP,
    )
    ax.grid(True, alpha=0.3)


def _mesh(X2: np.ndarray, step: int = 180):
    pad = 0.7
    xs = np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + pad, step)
    ys = np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + pad, step)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def _scatter_classes(ax, X2, y, class_names, size=18, alpha=0.9) -> None:
    for label, name in enumerate(class_names):
        mask = y == label
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            s=size,
            color=CLASS_COLORS[label],
            edgecolors="white",
            linewidths=0.3,
            alpha=alpha,
            label=name,
        )


def plot_boundaries_knn(X2, y, class_names, k: int) -> None:
    xx, yy, grid = _mesh(X2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    panels = (
        (axes[0], "parzen", f"Окно Парзена, k={k}"),
        (axes[1], "uniform", f"Равномерный KNN, k={k}"),
    )
    for ax, mode, title in panels:
        pred = predict(X2, y, grid, k, mode=mode)
        _regions(ax, xx, yy, pred, xx.shape)
        _scatter_classes(ax, X2, y, class_names)
        ax.set_title(title)
        ax.set_xlabel("1-я главная компонента")
        ax.set_ylabel("2-я главная компонента")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Разделяющая поверхность в плоскости главных компонент", y=1.02)
    fig.tight_layout()
    save_fig(fig, "05_boundary_knn")


def plot_prototypes(X2, y, roles, class_names) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6))
    for role, marker in ROLE_MARKERS.items():
        for label in range(len(class_names)):
            mask = (roles == role) & (y == label)
            if not np.any(mask):
                continue
            ax.scatter(
                X2[mask, 0],
                X2[mask, 1],
                s=ROLE_SIZES[role],
                marker=marker,
                color=CLASS_COLORS[label],
                linewidths=0.6,
                edgecolors="black" if role == "эталон" else "none",
                zorder=3 if role == "эталон" else 2,
            )
    class_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[i], markeredgecolor="black", markersize=8, label=name)
        for i, name in enumerate(class_names)
    ]
    role_handles = [
        Line2D([0], [0], marker=ROLE_MARKERS[role], color="none", markerfacecolor="0.2", markeredgecolor="black", markersize=8, label=role)
        for role in ROLE_MARKERS
        if np.any(roles == role)
    ]
    ax.legend(handles=class_handles + role_handles, fontsize=9)
    ax.set_xlabel("1-я главная компонента")
    ax.set_ylabel("2-я главная компонента")
    ax.set_title("Эталоны, неинформативные объекты и шум")
    fig.tight_layout()
    save_fig(fig, "06_prototypes")


def plot_boundaries_prototypes(X2, y, mask, class_names) -> None:
    xx, yy, grid = _mesh(X2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    panels = (
        (axes[0], np.ones(len(y), dtype=bool), "1NN на всей выборке"),
        (axes[1], mask, "1NN только на эталонах"),
    )
    for ax, train_mask, title in panels:
        pred = predict_1nn(X2[train_mask], y[train_mask], grid)
        _regions(ax, xx, yy, pred, xx.shape)
        faded = not np.all(train_mask)
        _scatter_classes(ax, X2, y, class_names, size=16, alpha=0.35 if faded else 0.95)
        if faded:
            ax.scatter(
                X2[train_mask, 0],
                X2[train_mask, 1],
                s=64,
                facecolors="none",
                edgecolors="black",
                linewidths=1.1,
                label="эталоны",
            )
        ax.set_title(title)
        ax.set_xlabel("1-я главная компонента")
        ax.set_ylabel("2-я главная компонента")
        ax.legend(fontsize=8)
    fig.suptitle("Поверхность 1NN до и после отбора эталонов", y=1.02)
    fig.tight_layout()
    save_fig(fig, "07_boundary_prototypes")


def plot_removal(history_removed, history_loo, history_kind, n_total: int) -> None:
    fig, ax = plt.subplots()
    ax.plot(history_removed, history_loo, color="black", linewidth=1.6, label="Q(Ω)")
    drops = [i for i, name in enumerate(history_kind) if name == "Q уменьшилось"]
    if drops:
        ax.scatter(
            [history_removed[i] for i in drops],
            [history_loo[i] for i in drops],
            color="tab:red",
            marker="o",
            s=36,
            zorder=3,
            label="удаление снизило Q",
        )
    left = n_total - history_removed[-1]
    ax.annotate(
        f"в Ω осталось {left}",
        xy=(history_removed[-1], history_loo[-1]),
        xytext=(-90, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    ax.set_xlabel("Число удалённых объектов")
    ax.set_ylabel("Эмпирический риск Q(Ω)")
    ax.set_title("Жадное удаление: риск 1NN по мере сжатия выборки")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "08_loo_removal")


def plot_comparison(names: list[str], loo_errors: list[float], test_errors: list[float]) -> None:
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars_loo = ax.bar(x - width / 2, loo_errors, width, label="ошибка LOO на всей выборке", color="tab:blue")
    bars_test = ax.bar(x + width / 2, test_errors, width, label="ошибка на отложенных 30%", color="tab:orange")
    ax.bar_label(bars_loo, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(bars_test, fmt="%.3f", fontsize=7, padding=2)
    top = max(loo_errors + test_errors)
    ax.set_ylim(0.0, top * 1.25 + 0.01)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Доля ошибок")
    ax.set_title("Сравнение алгоритмов")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "09_comparison")


def plot_confusion(y_test, panels: list[tuple[str, np.ndarray]], class_names: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, pred) in zip(axes, panels):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            pred,
            display_labels=class_names,
            ax=ax,
            colorbar=False,
        )
        acc = accuracy_score(y_test, pred)
        ax.set_title(f"{title}\nдоля верных {acc:.3f}")
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    save_fig(fig, "10_confusion")


def parzen_subset_search(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Подбор k заново: ширина окна, найденная на полной выборке, для маленького Ω слишком велика."""
    n_proto = int(np.sum(mask))
    best_k = 1
    best_pred, _ = predict_loo_parzen_subset(X, y, mask, best_k)
    best_error = float(np.mean(best_pred != y))
    for k in range(2, n_proto - 1):
        pred, _ = predict_loo_parzen_subset(X, y, mask, k)
        error = float(np.mean(pred != y))
        if error < best_error:
            best_k = k
            best_error = error
            best_pred = pred
    return best_k, best_error, best_pred


def curve_stats(ks: np.ndarray, risks: np.ndarray) -> dict:
    rmin = float(np.min(risks))
    at_min = np.where(np.isclose(risks, rmin, atol=1e-15))[0]
    picks = (1, 5, 10, 15, 20, 31, 36, 50, 80, 100, 150)
    known = set(int(k) for k in ks)
    return {
        "k_first_min": int(ks[at_min[0]]),
        "k_last_min": int(ks[at_min[-1]]),
        "n_at_min": int(len(at_min)),
        "risk_min": rmin,
        "risk_at": {str(k): float(risks[ks == k][0]) for k in picks if k in known},
    }


def best_k(ks: np.ndarray, risks: np.ndarray) -> tuple[int, float]:
    index = int(np.argmin(risks))
    return int(ks[index]), float(risks[index])


def pack_curve(ks: np.ndarray, risks: np.ndarray) -> dict[str, float]:
    """Несколько точек кривой для отчёта, не вся сетка."""
    picks = {
        "k1": (int(ks[0]), float(risks[0])),
        "k_min": best_k(ks, risks),
        "k_max": (int(ks[-1]), float(risks[-1])),
    }
    out = {}
    for name, (k, risk) in picks.items():
        out[f"{name}"] = k
        out[f"{name}_risk"] = risk
    return out


def main() -> None:
    self_check()
    data = load_wine_data()
    X_std, _ = standardize(data.X)
    y = data.y
    dist = loo_distance_matrix(X_std)

    print("LOO по сетке k...")
    ks_parzen, risks_parzen = loo_curve(dist, y, mode="parzen")
    ks_uniform, risks_uniform = loo_curve(dist, y, mode="uniform")
    k_star, _ = best_k(ks_parzen, risks_parzen)
    k_uniform, _ = best_k(ks_uniform, risks_uniform)

    print("сверка со sklearn и замер времени...")

    def our_loo():
        matrix = loo_distance_matrix(X_std)
        return loo_predictions(matrix, y, k_star, mode="parzen")

    def lib_loo():
        return sklearn_loo(X_std, y, k_star)

    time_our, pred_our = time_call(our_loo, repeats=5)
    time_lib, pred_lib = time_call(lib_loo, repeats=3)
    pred_uniform = loo_predictions(dist, y, k_star, mode="uniform")
    sklearn_mismatch = int(np.sum(pred_lib != pred_uniform))

    print("отбор эталонов...")
    started = time.perf_counter()
    selected = greedy_condense(dist, y)
    selection_time = time.perf_counter() - started

    roles = lecture_roles(selected.mask, selected.margins)
    pred_1nn = predict_loo_1nn(dist, y)
    pred_1nn_proto = predict_loo_1nn(dist, y, selected.mask)
    k_proto, _, pred_parzen_proto = parzen_subset_search(X_std, y, selected.mask)
    pred_uniform_best = loo_predictions(dist, y, k_uniform, mode="uniform")
    pred_lib_own = sklearn_loo(X_std, y, k_uniform)
    sklearn_mismatch_own = int(np.sum(pred_lib_own != pred_uniform_best))

    majority = int(np.bincount(y).argmax())
    majority_error = float(np.mean(y != majority))

    plot_dataset(data.X, y, data.feature_names_ru, data.class_names)
    plot_margins(selected.margins, roles)
    plot_loo(ks_parzen, risks_parzen, ks_uniform, risks_uniform, majority_error)
    plot_kernel()

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_std)
    plot_boundaries_knn(X2, y, data.class_names, k_star)
    plot_prototypes(X2, y, roles, data.class_names)
    plot_boundaries_prototypes(X2, y, selected.mask, data.class_names)
    plot_removal(selected.history_removed, selected.history_loo, selected.history_kind, len(y))

    print("отложенная выборка...")
    X_train, X_test, y_train, y_test = stratified_split(data)
    dist_train = loo_distance_matrix(X_train)
    ks_tr_p, risks_tr_p = loo_curve(dist_train, y_train, mode="parzen")
    ks_tr_u, risks_tr_u = loo_curve(dist_train, y_train, mode="uniform")
    k_train, _ = best_k(ks_tr_p, risks_tr_p)
    k_train_uniform, _ = best_k(ks_tr_u, risks_tr_u)
    def run_train_selection():
        return greedy_condense(dist_train, y_train)

    selection_train_time, selected_train = time_call(run_train_selection, repeats=3)
    roles_train = lecture_roles(selected_train.mask, selected_train.margins)
    k_train_proto, _, _ = parzen_subset_search(X_train, y_train, selected_train.mask)

    pred_test = {
        "parzen": predict(X_train, y_train, X_test, k_train, mode="parzen"),
        "sklearn_same_k": KNeighborsClassifier(
            n_neighbors=k_train, weights="uniform", metric="euclidean", algorithm="brute"
        ).fit(X_train, y_train).predict(X_test),
        "sklearn_own_k": KNeighborsClassifier(
            n_neighbors=k_train_uniform, weights="uniform", metric="euclidean", algorithm="brute"
        ).fit(X_train, y_train).predict(X_test),
        "nn_all": predict_1nn(X_train, y_train, X_test),
        "nn_proto": predict_1nn(X_train[selected_train.mask], y_train[selected_train.mask], X_test),
        "parzen_proto": predict(
            X_train[selected_train.mask],
            y_train[selected_train.mask],
            X_test,
            k_train_proto,
            mode="parzen",
        ),
    }

    def time_1nn_predict(X_ref: np.ndarray, y_ref: np.ndarray, inner: int = 200) -> float:
        """Медианное время одного предсказания 54 объектов. Внутри замера — пачка вызовов, чтобы фиксированная стоимость вызова не съела разницу."""
        predict_1nn(X_ref, y_ref, X_test)

        def batch():
            out = None
            for _ in range(inner):
                out = predict_1nn(X_ref, y_ref, X_test)
            return out

        seconds, _ = time_call(batch, repeats=7)
        return seconds / inner

    proto_ref = selected_train.mask
    time_1nn_full = time_1nn_predict(X_train, y_train)
    time_1nn_proto = time_1nn_predict(X_train[proto_ref], y_train[proto_ref])
    distances_full = int(len(X_test) * len(X_train))
    distances_proto = int(len(X_test) * int(proto_ref.sum()))
    bytes_full = int(X_train.nbytes + y_train.nbytes)
    bytes_proto = int(X_train[proto_ref].nbytes + y_train[proto_ref].nbytes)

    def time_predict(mode: str, k: int) -> float:
        seconds, _ = time_call(lambda: predict(X_train, y_train, X_test, k, mode=mode), repeats=20)
        return seconds

    def time_sklearn_predict(k: int) -> float:
        def run():
            model = KNeighborsClassifier(
                n_neighbors=k, weights="uniform", metric="euclidean", algorithm="brute"
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)

        seconds, _ = time_call(run, repeats=20)
        return seconds

    loo_rows = {
        "Парзен": quality(y, pred_our),
        "sklearn, тот же k": quality(y, pred_lib),
        "sklearn, свой k": quality(y, pred_uniform_best),
        "1NN, вся выборка": quality(y, pred_1nn),
        "1NN, эталоны": quality(y, pred_1nn_proto),
        "Парзен, эталоны": quality(y, pred_parzen_proto),
    }
    # Предсказания sklearn при своём k берём из нашей матрицы: ниже проверяем,
    # что при общем k библиотека совпала с этой матрицей.
    loo_rows["sklearn, свой k"] = quality(y, pred_lib_own)

    test_rows = {name: quality(y_test, pred) for name, pred in pred_test.items()}
    order = [
        ("Парзен", "parzen"),
        ("sklearn, тот же k", "sklearn_same_k"),
        ("sklearn, свой k", "sklearn_own_k"),
        ("1NN, вся выборка", "nn_all"),
        ("1NN, эталоны", "nn_proto"),
        ("Парзен, эталоны", "parzen_proto"),
    ]
    plot_comparison(
        [name for name, _ in order],
        [loo_rows[name]["error"] for name, _ in order],
        [test_rows[key]["error"] for _, key in order],
    )
    plot_confusion(
        y_test,
        [
            (f"Парзен, k={k_train}", pred_test["parzen"]),
            (f"sklearn, k={k_train_uniform}", pred_test["sklearn_own_k"]),
            ("1NN на эталонах", pred_test["nn_proto"]),
        ],
        data.class_names,
    )

    role_counts = {role: int(np.sum(roles == role)) for role in ("эталон", "неинформативный", "шум")}
    margin_by_role = {}
    for role in role_counts:
        values = selected.margins[roles == role]
        margin_by_role[role] = {
            "mean": float(values.mean()) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }

    summary = {
        "dataset": {
            "name": "Wine",
            "n": int(len(y)),
            "n_features": int(X_std.shape[1]),
            "class_counts": {data.class_names[i]: int(np.sum(y == i)) for i in range(3)},
            "n_missing": int(np.isnan(data.X).sum()),
            "feature_std": {
                data.feature_names_ru[i]: float(data.X.std(axis=0)[i]) for i in range(data.X.shape[1])
            },
            "feature_min": {
                data.feature_names_ru[i]: float(data.X.min(axis=0)[i]) for i in range(data.X.shape[1])
            },
            "feature_max": {
                data.feature_names_ru[i]: float(data.X.max(axis=0)[i]) for i in range(data.X.shape[1])
            },
            "pca_explained": [float(v) for v in pca.explained_variance_ratio_],
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "majority_error": majority_error,
            "n_negative_margin": int(np.sum(selected.margins < 0)),
        },
        "loo_full": {
            "parzen": pack_curve(ks_parzen, risks_parzen),
            "uniform": pack_curve(ks_uniform, risks_uniform),
            "parzen_shape": curve_stats(ks_parzen, risks_parzen),
            "uniform_shape": curve_stats(ks_uniform, risks_uniform),
            "k_star": k_star,
            "k_uniform": k_uniform,
            "sklearn_mismatch_at_k_star": sklearn_mismatch,
            "sklearn_mismatch_at_own_k": sklearn_mismatch_own,
            "time_parzen_loo_s": time_our,
            "time_sklearn_loo_s": time_lib,
            "time_predict_parzen_s": time_predict("parzen", k_train),
            "time_predict_uniform_s": time_predict("uniform", k_train),
            "time_predict_sklearn_s": time_sklearn_predict(k_train),
            "rows": loo_rows,
        },
        "prototypes": {
            "counts": role_counts,
            "margin_by_role": margin_by_role,
            "loo_before": selected.history_loo[0],
            "loo_after": selected.history_loo[-1],
            "removed": len(selected.removed_order),
            "selection_time_s": selection_time,
            "k_proto_full": k_proto,
            "q_decreased_steps": int(sum(kind == "Q уменьшилось" for kind in selected.history_kind)),
            "negative_margin_kept": int(np.sum(selected.mask & (selected.margins < 0))),
            "classes_in_omega": [int(c) for c in np.unique(y[selected.mask])],
        },
        "holdout": {
            "k_parzen": k_train,
            "k_uniform": k_train_uniform,
            "k_parzen_proto": k_train_proto,
            "n_prototypes": int(selected_train.mask.sum()),
            "role_counts": {
                role: int(np.sum(roles_train == role))
                for role in ("эталон", "неинформативный", "шум")
            },
            "loo_train_1nn_before": selected_train.history_loo[0],
            "loo_train_1nn_after": selected_train.history_loo[-1],
            "train_parzen_shape": curve_stats(ks_tr_p, risks_tr_p),
            "train_uniform_shape": curve_stats(ks_tr_u, risks_tr_u),
            "rows": test_rows,
            "loo_train_parzen": float(risks_tr_p.min()),
            "loo_train_uniform": float(risks_tr_u.min()),
            "selection_train_time_s": selection_train_time,
            "time_1nn_test_full_s": time_1nn_full,
            "time_1nn_test_proto_s": time_1nn_proto,
            "distances_full": distances_full,
            "distances_proto": distances_proto,
            "bytes_full": bytes_full,
            "bytes_proto": bytes_proto,
        },
    }
    IMAGE_DIR.mkdir(exist_ok=True)
    (IMAGE_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
