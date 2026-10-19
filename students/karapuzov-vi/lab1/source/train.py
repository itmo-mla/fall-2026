"""Пункты 9–11: три протокола обучения, метрики, сравнение со sklearn."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier, SGDClassifier

from data import prepare_data, PLOTS_DIR
from model import (
    add_bias,
    classify_metrics,
    empirical_risk,
    init_correlation,
    margins,
    plot_margins,
    predict,
    sgd_momentum,
    train_multistart,
)

SGD_PARAMS = dict(h=0.001, tau=0.9, n_steps=8000, l2=0.1, steepest=False)


def print_metrics(name: str, split: str, metrics: dict) -> None:
    print(
        f"{name:28} {split:5}  "
        f"acc={metrics['accuracy']*100:5.2f}%  "
        f"err={metrics['error']*100:5.2f}%  "
        f"P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}"
    )


def evaluate_model(name, w, X_train, y_train, X_test, y_test) -> dict:
    pred_tr = predict(X_train, w)
    pred_te = predict(X_test, w)
    m_tr = classify_metrics(y_train, pred_tr)
    m_te = classify_metrics(y_test, pred_te)
    print_metrics(name, "train", m_tr)
    print_metrics(name, "test", m_te)
    return {"name": name, "w": w, "train": m_tr, "test": m_te, "pred_test": pred_te}


def plot_confusion(y_true, y_pred, title: str, filename: str) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.array(
        [
            [np.sum((y_pred == -1) & (y_true == -1)), np.sum((y_pred == 1) & (y_true == -1))],
            [np.sum((y_pred == -1) & (y_true == 1)), np.sum((y_pred == 1) & (y_true == 1))],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["pred −1", "pred +1"])
    ax.set_yticks([0, 1], labels=["true −1", "true +1"])
    ax.set_title(title)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", color="black")
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=120)
    plt.close(fig)


def plot_f1_bars(results: list[dict], filename: str) -> None:
    names = [r["name"] for r in results]
    f1_tr = [r["train"]["f1"] for r in results]
    f1_te = [r["test"]["f1"] for r in results]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f1_tr, width, label="F1 train")
    ax.bar(x + width / 2, f1_te, width, label="F1 test")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("F1 (класс +1, выгорание)")
    ax.set_title("Сравнение протоколов обучения и эталона")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    data = prepare_data()
    X_train = add_bias(data["X_train"])
    X_test = add_bias(data["X_test"])
    y_train = np.asarray(data["y_train"], dtype=int)
    y_test = np.asarray(data["y_test"], dtype=int)
    X_train_sk = np.asarray(data["X_train"], dtype=float)
    X_test_sk = np.asarray(data["X_test"], dtype=float)

    w_corr = init_correlation(X_train, y_train)
    results = []

    print("=== пункт 9: три протокола ===")
    w1, _ = sgd_momentum(X_train, y_train, w_corr, sampling="uniform", **SGD_PARAMS)
    results.append(evaluate_model("9.1 corr + uniform", w1, X_train, y_train, X_test, y_test))

    w2, hist_m = sgd_momentum(X_train, y_train, w_corr, sampling="margin", **SGD_PARAMS)
    results.append(evaluate_model("9.3 corr + |M|", w2, X_train, y_train, X_test, y_test))
    plot_margins(margins(X_train, y_train, w2), "Отступы train: corr + |M|", "final_margins_train.png")

    w3, _, q3 = train_multistart(X_train, y_train, n_starts=5, sampling="uniform", **SGD_PARAMS)
    print(f"9.2 мультистарт: лучший Q(train)={q3:.4f}")
    results.append(evaluate_model("9.2 random multistart", w3, X_train, y_train, X_test, y_test))

    w4, _, q4 = train_multistart(X_train, y_train, n_starts=5, sampling="margin", **SGD_PARAMS)
    print(f"9.2+8 мультистарт+|M|: лучший Q(train)={q4:.4f}")
    results.append(evaluate_model("multistart + |M|", w4, X_train, y_train, X_test, y_test))

    print("\n=== пункт 11: эталон sklearn ===")
    ridge = RidgeClassifier()
    ridge.fit(X_train_sk, y_train)
    ridge_w = np.r_[ridge.intercept_, ridge.coef_.ravel()]
    results.append(evaluate_model("sklearn RidgeClassifier", ridge_w, X_train, y_train, X_test, y_test))

    sgd_sk = SGDClassifier(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        max_iter=8000,
        tol=None,
        random_state=0,
        learning_rate="constant",
        eta0=0.001,
    )
    sgd_sk.fit(X_train_sk, y_train)
    sgd_w = np.r_[sgd_sk.intercept_, sgd_sk.coef_.ravel()]
    results.append(evaluate_model("sklearn SGDClassifier", sgd_w, X_train, y_train, X_test, y_test))

    best = max(results[:4], key=lambda r: r["test"]["f1"])
    print("\nлучший среди наших реализаций по F1 test:", best["name"], f"(F1={best['test']['f1']:.3f})")
    plot_confusion(
        y_test,
        best["pred_test"],
        f"Матрица ошибок test: {best['name']}",
        "confusion_best_test.png",
    )
    plot_confusion(
        y_test,
        results[4]["pred_test"],
        "Матрица ошибок test: RidgeClassifier",
        "confusion_ridge_test.png",
    )
    plot_f1_bars(results, "f1_comparison.png")
    print("графики:", PLOTS_DIR)
