"""Эксперименты лабораторной №1: обучение, визуализация, сравнение с эталоном."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
IMAGE_DIR = ROOT / "image"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from dataset import load_binary_dataset
from linear_classifier import (
    LinearSGDClassifier,
    empirical_risk,
    epochs_until_risk,
    init_correlation,
    init_random,
    margins,
    multistart,
    weight_norm,
)

plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "font.size": 11,
    }
)

L2 = 1e-3
POS_LABEL = 1


def save_fig(fig, name: str) -> Path:
    IMAGE_DIR.mkdir(exist_ok=True)
    path = IMAGE_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def print_split_metrics(title: str, table: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    for name, m in table.items():
        print(
            f"  {name:22s} acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}"
        )


def plot_dataset_overview(meta: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].bar(
        ["злокач. (−1)", "доброкач. (+1)"],
        [meta["n_neg_full"], meta["n_pos_full"]],
        color=["tab:orange", "tab:blue"],
        label="все объекты",
    )
    axes[0].set_title("Баланс классов")
    axes[0].set_ylabel("Число объектов")
    axes[0].legend()

    n_show = 10
    names = [n.replace(" ", "\n") for n in meta["feature_names"][:n_show]]
    axes[1].bar(range(n_show), meta["feature_max"][:n_show] - meta["feature_min"][:n_show])
    axes[1].set_xticks(range(n_show))
    axes[1].set_xticklabels(names, fontsize=7)
    axes[1].set_title("Размах первых 10 признаков (до стандартизации)")
    axes[1].set_ylabel("max − min")

    corr = np.abs(meta["corr_train"])
    im = axes[2].imshow(corr, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("|корреляция| признаков на train")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    save_fig(fig, "01_dataset")


def plot_margins(M: np.ndarray, title: str, filename: str) -> None:
    M_sorted = np.sort(M)
    xs = np.arange(1, len(M_sorted) + 1)
    fig, ax = plt.subplots()
    ax.plot(xs, M_sorted, "-", color="0.55", linewidth=1, label="отступы по возрастанию")
    ok = M_sorted >= 0
    err = M_sorted < 0
    ax.plot(xs[ok], M_sorted[ok], "o", markersize=3, color="tab:blue", label="M ≥ 0 (верно)")
    if np.any(err):
        ax.plot(xs[err], M_sorted[err], "o", markersize=4, color="tab:orange", label="M < 0 (ошибка)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="граница M = 0")
    ax.set_xlabel("Количество объектов (порядковый номер)")
    ax.set_ylabel("Отступ M = y ⟨w, x⟩")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, filename)


def plot_histories(
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    filename: str,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots()
    for name, values in series.items():
        ax.plot(values, marker="o", markersize=3, linewidth=1, label=name)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, filename)


def plot_pca_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X[:, 1:])
    fig, ax = plt.subplots()
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=18, label="класс +1 (доброкач.)", alpha=0.8)
    ax.scatter(Z[y == -1, 0], Z[y == -1, 1], s=18, label="класс −1 (злокач.)", alpha=0.8)

    x_min, x_max = Z[:, 0].min() - 1, Z[:, 0].max() + 1
    y_min, y_max = Z[:, 1].min() - 1, Z[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid = pca.inverse_transform(grid)
    X_grid = np.hstack([np.ones((X_grid.shape[0], 1)), X_grid])
    zz = np.sign(X_grid @ w).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-1, 0, 1], alpha=0.15, colors=["#d62728", "#1f77b4"])
    ax.set_xlabel("1-я главная компонента")
    ax.set_ylabel("2-я главная компонента")
    ax.set_title("Разделяющая поверхность в проекции на две главные компоненты")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "07_pca")


def plot_train_test_bars(
    train_table: dict[str, dict[str, float]],
    test_table: dict[str, dict[str, float]],
    metric_name: str,
    title: str,
    filename: str,
) -> None:
    names = list(train_table.keys())
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [train_table[n][metric_name] for n in names], width, label="обучение")
    ax.bar(x + width / 2, [test_table[n][metric_name] for n in names], width, label="тест")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, filename)


def plot_confusion(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.2))
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["злокач. −1", "доброкач. +1"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    fig.tight_layout()
    save_fig(fig, "11_confusion")


def plot_roc(y_true: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots()
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score, pos_label=POS_LABEL)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}, AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="случайный классификатор")
    ax.set_xlabel("Доля ложных срабатываний")
    ax.set_ylabel("Доля истинных срабатываний")
    ax.set_title("Кривые ошибок (ROC) на тесте")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "12_roc")


def plot_l2_sweep(taus: list[float], norms: list[float], acc_test: list[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(taus, norms, marker="o", linewidth=1, label="норма весов без свободного члена")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("коэффициент штрафа tau")
    axes[0].set_ylabel("||w||")
    axes[0].set_title("Норма весов при росте L2")
    axes[0].legend()
    axes[1].plot(taus, acc_test, marker="o", linewidth=1, color="tab:orange", label="доля верных на тесте")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("коэффициент штрафа tau")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Качество на тесте при росте L2")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "08_l2")


def model_scores(model: LinearSGDClassifier, X: np.ndarray) -> np.ndarray:
    return model.decision_function(X)


def main() -> None:
    X_train, X_test, y_train, y_test, meta = load_binary_dataset()
    n_features = meta["n_features"]
    ridge_alpha = L2 * len(y_train) / 2.0

    print(f"Датасет: {meta['name']}")
    print(f"Источник: {meta['source']}")
    print(f"объектов: {meta['n_pos_full'] + meta['n_neg_full']}, признаков: {n_features}")
    print(f"пропусков: {meta['n_missing']}")
    print(
        f"классы (все): доброкач. {meta['n_pos_full']} "
        f"({meta['n_pos_full'] / (meta['n_pos_full'] + meta['n_neg_full']):.1%}), "
        f"злокач. {meta['n_neg_full']}"
    )
    print(
        f"train={meta['n_train']} (+:{meta['n_pos_train']}, -:{meta['n_neg_train']}), "
        f"test={meta['n_test']} (+:{meta['n_pos_test']}, -:{meta['n_neg_test']})"
    )
    print(
        f"пар признаков с |корреляцией| > 0.9 на train: {meta['n_high_corr_pairs']}, "
        f"средняя |корреляция|: {meta['mean_abs_corr']:.3f}"
    )
    print(f"штраф эталона alpha={ridge_alpha:.4f} подобран под l2={L2} и n={len(y_train)}")
    plot_dataset_overview(meta)

    rng = np.random.default_rng(0)
    w_rand = init_random(X_train.shape[1], rng, n_features=n_features)
    w_corr = init_correlation(X_train, y_train)
    plot_margins(
        margins(X_train, y_train, w_corr),
        "Отступы после корреляционной инициализации",
        "02_margins_corr",
    )
    plot_margins(
        margins(X_train, y_train, w_rand),
        "Отступы после случайной инициализации",
        "03_margins_random",
    )

    common = dict(l2=L2, lr=0.2, momentum=0.9, n_epochs=60, smooth=0.05)

    model_plain = LinearSGDClassifier(
        sampling="random", step="constant", random_state=7, l2=L2, lr=0.2, momentum=0.0, n_epochs=60, smooth=0.05
    )
    res_plain = model_plain.fit(X_train, y_train, w0=w_corr)

    model_random = LinearSGDClassifier(sampling="random", step="constant", random_state=7, **common)
    res_random = model_random.fit(X_train, y_train, w0=w_corr)

    model_margin = LinearSGDClassifier(sampling="margin", step="constant", random_state=7, **common)
    res_margin = model_margin.fit(X_train, y_train, w0=w_corr)

    model_steep = LinearSGDClassifier(
        sampling="random",
        step="steepest",
        l2=L2,
        momentum=0.0,
        n_epochs=40,
        smooth=0.05,
        random_state=3,
    )
    res_steep = model_steep.fit(X_train, y_train, w0=w_corr)

    model_multi, multi_risks = multistart(
        X_train,
        y_train,
        n_starts=8,
        random_state=42,
        sampling="random",
        step="constant",
        **common,
    )

    plot_margins(
        margins(X_train, y_train, model_random.w_),
        "Отступы после обучения SGD",
        "04_margins_sgd",
    )
    plot_histories(
        {
            "SGD без инерции": res_plain.risk_history,
            "SGD с инерцией": res_random.risk_history,
        },
        "Эмпирический риск Q(w)",
        "Сходимость: обычный SGD и SGD с инерцией",
        "05_inertia",
        log_y=True,
    )
    plot_histories(
        {
            "случайное предъявление": res_random.q_history,
            "предъявление пограничных объектов": res_margin.q_history,
            "мультистарт (лучший)": model_multi.q_history_,
            "скорейший спуск": res_steep.q_history,
        },
        "Оценка средней потери",
        "Рекуррентная оценка функционала",
        "06_q_hat",
    )
    plot_histories(
        {
            "случайное предъявление": res_random.risk_history,
            "предъявление пограничных объектов": res_margin.risk_history,
            "мультистарт (лучший)": model_multi.risk_history_,
            "скорейший спуск": res_steep.risk_history,
        },
        "Эмпирический риск Q(w)",
        "Эмпирический риск на обучающей выборке",
        "06_q_risk",
        log_y=True,
    )
    plot_pca_boundary(X_train, y_train, model_random.w_)

    taus = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    l2_norms: list[float] = []
    l2_acc: list[float] = []
    for tau in taus:
        model_tau = LinearSGDClassifier(
            sampling="random",
            step="steepest",
            l2=tau,
            momentum=0.0,
            n_epochs=40,
            random_state=3,
        )
        model_tau.fit(X_train, y_train, w0=w_corr)
        l2_norms.append(weight_norm(model_tau.w_))
        l2_acc.append(accuracy_score(y_test, model_tau.predict(X_test)))
    plot_l2_sweep(taus, l2_norms, l2_acc)
    print("\nL2-штраф (скорейший спуск):")
    for tau, nrm, acc in zip(taus, l2_norms, l2_acc):
        print(f"  tau={tau:.0e}  ||w||={nrm:.3f}  acc_test={acc:.4f}")

    ridge = RidgeClassifier(alpha=ridge_alpha)
    ridge.fit(X_train[:, 1:], y_train)

    own = {
        "SGD без инерции": model_plain,
        "SGD с инерцией": model_random,
        "пограничные объекты": model_margin,
        "мультистарт": model_multi,
        "скорейший спуск": model_steep,
    }
    histories = {
        "SGD без инерции": res_plain.risk_history,
        "SGD с инерцией": res_random.risk_history,
        "пограничные объекты": res_margin.risk_history,
        "мультистарт": model_multi.risk_history_,
        "скорейший спуск": res_steep.risk_history,
    }
    train_risks = {name: empirical_risk(X_train, y_train, m.w_, m.l2) for name, m in own.items()}
    best_name = min(train_risks, key=train_risks.get)
    best_model = own[best_name]

    def pack(split_X, split_y):
        table = {}
        for name, model in own.items():
            scores = model_scores(model, split_X)
            table[name] = metrics(split_y, model.predict(split_X), scores)
        ridge_scores = ridge.decision_function(split_X[:, 1:])
        table["эталон Ridge"] = metrics(split_y, ridge.predict(split_X[:, 1:]), ridge_scores)
        return table

    train_table = pack(X_train, y_train)
    test_table = pack(X_test, y_test)

    plot_train_test_bars(train_table, test_table, "accuracy", "Доля верных: обучение против теста", "09_acc")
    plot_train_test_bars(train_table, test_table, "roc_auc", "Площадь под ROC: обучение против теста", "10_auc")
    plot_confusion(
        y_test,
        {
            f"лучшая своя ({best_name})": best_model.predict(X_test),
            "эталон Ridge": ridge.predict(X_test[:, 1:]),
        },
    )
    plot_roc(
        y_test,
        {
            best_name: model_scores(best_model, X_test),
            "эталон Ridge": ridge.decision_function(X_test[:, 1:]),
        },
    )

    print("\nРиск Q на обучении:")
    for name, risk in train_risks.items():
        print(f"  {name:22s} {risk:.6f}")
    print(f"Мультистарт, риски стартов: {np.round(multi_risks, 5).tolist()}")
    print(f"Лучшая своя модель по Q на обучении: {best_name}")
    print_split_metrics("Метрики на обучении:", train_table)
    print_split_metrics("Метрики на тесте:", test_table)

    print("\nНорма весов и скорость (эпох до Q <= 0.5):")
    print(f"  {'модель':22s}  ||w||      эпох до Q<=0.5  эпох всего")
    for name, model in own.items():
        print(
            f"  {name:22s}  {weight_norm(model.w_):8.3f}  "
            f"{epochs_until_risk(histories[name]):16d}  {len(model.q_history_) - 1:10d}"
        )
    ridge_norm = float(np.linalg.norm(np.ravel(ridge.coef_)))
    print(f"  {'эталон Ridge':22s}  {ridge_norm:8.3f}                --          --")
    print(
        f"Инерция: Q<=0.5 без неё на эпохе {epochs_until_risk(res_plain.risk_history)}, "
        f"с ней на эпохе {epochs_until_risk(res_random.risk_history)}"
    )

    n_err_init = int(np.sum(margins(X_train, y_train, w_corr) < 0))
    n_err_fit = int(np.sum(margins(X_train, y_train, model_random.w_) < 0))
    print(f"\nОшибок на обучении (корреляционный старт): {n_err_init} -> {n_err_fit}")
    print(f"Средний отступ при старте: {margins(X_train, y_train, w_corr).mean():.3f}")
    print(f"Средний отступ после SGD: {margins(X_train, y_train, model_random.w_).mean():.3f}")
    print(f"Графики сохранены в {IMAGE_DIR}")


if __name__ == "__main__":
    main()
