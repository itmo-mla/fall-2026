"""
Задача 8: обучить линейный классификатор на выбранном датасете:
  8.1 - с инициализацией весов через корреляцию;
  8.2 - со случайной инициализацией весов через мультистарт;
  8.3 - со случайным предъявлением и с margin-based предъявлением (п.7).

Подключает task07 (and all task01...task06).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from task07_margin_presentation import LinearClassifier, load_and_prepare, add_bias
from task01_margin import correlation_init

BASE = dict(eta=0.01, gamma=0.9, n_epochs=10, tau=0.0, step_mode="fixed")


def evaluate(clf, X_test, y_test, name):
    y_pred = clf.predict(X_test)
    return {
        "name": name,
        "Q_final": clf.Q_history[-1],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare()
    Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

    results = []

    # --- 8.1 Корреляционная инициализация ---
    w_corr = correlation_init(Xb_train, y_train)
    clf_corr = LinearClassifier(presentation="uniform", random_state=42, **BASE)
    clf_corr.fit(Xb_train, y_train, w_init=w_corr)
    results.append(evaluate(clf_corr, Xb_test, y_test, "8.1 Корреляционная инициализация"))

    # --- 8.2 Мультистарт (K случайных инициализаций, выбор лучшей по Q) ---
    K = 5
    best_clf, best_Q = None, np.inf
    rng = np.random.RandomState(0)
    q_runs = []
    for k in range(K):
        w0 = rng.normal(scale=0.01, size=Xb_train.shape[1])
        clf_k = LinearClassifier(presentation="uniform", random_state=100 + k, **BASE)
        clf_k.fit(Xb_train, y_train, w_init=w0)
        q_runs.append(clf_k.Q_history[-1])
        if clf_k.Q_history[-1] < best_Q:
            best_Q, best_clf = clf_k.Q_history[-1], clf_k
    print("Q_final по всем стартам мультистарта:", np.round(q_runs, 4))
    results.append(evaluate(best_clf, Xb_test, y_test, "8.2 Мультистарт (лучший из K=5)"))

    # --- 8.3 Равновероятное vs margin-based предъявление ---
    clf_uniform = LinearClassifier(presentation="uniform", random_state=42, **BASE)
    clf_uniform.fit(Xb_train, y_train)
    results.append(evaluate(clf_uniform, Xb_test, y_test, "8.3a Равновероятное предъявление"))

    clf_margin = LinearClassifier(presentation="margin", random_state=42, **BASE)
    clf_margin.fit(Xb_train, y_train)
    results.append(evaluate(clf_margin, Xb_test, y_test, "8.3b Margin-based предъявление"))

    print(f"\n{'Вариант':40s} {'Q_final':>8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
    for r in results:
        print(f"{r['name']:40s} {r['Q_final']:8.4f} {r['accuracy']:7.4f} "
              f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}")

    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(results))
    width = 0.2

    # fig, ax = plt.subplots(figsize=(9, 4.5))
    # names = [r["name"].replace(" ", "\n", 1) for r in results]
    # f1_vals = [r["f1"] for r in results]
    # bars = ax.bar(names, f1_vals, color=["steelblue", "steelblue", "gray", "darkorange"])
    # ax.set_ylabel("F1 (класс 'отказ')")
    # ax.bar_label(bars, fmt="%.3f", padding=3)
    # ax.set_title("Задача 8: сравнение вариантов обучения")
    # ax.legend(handles=bars.patches[:1], labels=["F1 на тестовой выборке"])

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        vals = [r[m] for r in results]
        bars = ax.bar(x + (i - 1.5)*width, vals, width, label=m.capitalize())
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in results], rotation=15, ha="right")
    ax.set_title("Задача 8: сравнение вариантов обучения")
    ax.legend()

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("../images/task08_train_variants.png", dpi=120)
    print("\nГрафик сохранён: task08_train_variants.png")

    print("\nАнализ: инициализация (корреляционная vs мультистарт) почти не "
          "влияет на итог при таком дисбалансе классов. Единственный вариант, "
          "реально сдвигающий классификатор от коллапса - margin-based "
          "предъявление (8.3b).")
