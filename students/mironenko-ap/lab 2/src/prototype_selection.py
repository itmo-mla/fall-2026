"""Отбор эталонов: CNN (Hart, 1968) и ENN (Wilson, 1972)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import load_raw, make_xy, train_test_split_np, standardize
from .parzen_knn import ParzenKNN
from .metrics import accuracy, macro_f1


# ============================================================
# 1. CNN — Condensed Nearest Neighbor (для 1-NN)
# ============================================================
def cnn_select(X, y):
    """
    Жадный отбор эталонов для 1-NN (Hart, 1968).

    Стартуем с одного объекта каждого класса. Добавляем точку,
    если текущий набор 1-NN классифицирует её неверно. Повторяем
    проходы до стабилизации.

    Возвращает индексы выбранных эталонов.
    """
    n = len(X)
    classes = np.unique(y)

    selected = [int(np.where(y == c)[0][0]) for c in classes]
    selected = np.array(sorted(set(selected)))

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i in selected:
                continue
            d = np.sqrt(((X[selected] - X[i]) ** 2).sum(axis=1))
            pred = y[selected[np.argmin(d)]]
            if pred != y[i]:
                selected = np.append(selected, i)
                changed = True
        selected = np.array(sorted(set(selected.tolist())))

    return selected


# ============================================================
# 2. ENN — Edited Nearest Neighbor (для любого k)
# ============================================================
def enn_select(X, y, k: int = 3):
    """
    Edited Nearest Neighbor (Wilson, 1972).

    Удаляет точку x_i, если класс большинства её k ближайших
    соседей не совпадает с y_i. Убирает шум, оставляет структуру
    классов.

    Возвращает индексы СОХРАНЁННЫХ точек.
    """
    n = len(X)
    diffs = X[:, None, :] - X[None, :, :]
    sq = np.einsum("ijk,ijk->ij", diffs, diffs)
    np.fill_diagonal(sq, np.inf)
    k_eff = min(k, n - 1)
    knn_idx = np.argpartition(sq, k_eff - 1, axis=1)[:, :k_eff]

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        neighbor_y = y[knn_idx[i]]
        counts = np.bincount(neighbor_y)
        majority = int(np.argmax(counts))
        if majority != y[i]:
            keep[i] = False
    return np.where(keep)[0]


# ============================================================
# 3. Оценка
# ============================================================
def _eval_model(X_tr, y_tr, X_te, y_te, n_classes, k):
    model = ParzenKNN(k=k).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "accuracy": accuracy(y_te, y_pred),
        "macro_f1": macro_f1(y_te, y_pred, n_classes),
        "n_train": len(X_tr),
    }


# ============================================================
# 4. Основной сценарий: сравнение full / CNN / ENN
# ============================================================
def run_selection(out_dir: Path,
                  k: int = 3,
                  enn_k: int = 3,
                  random_state: int = 42):
    """
    Сравнивает ParzenKNN(k) на трёх вариантах обучающего набора:
        - полный train
        - CNN-эталоны
        - ENN-эталоны

    Сохраняет:
        report/figures/prototypes_cnn_idx.npy
        report/figures/prototypes_enn_idx.npy
        report/figures/selection_compare.csv
        report/figures/selection_compare.png
    """
    data = load_raw()
    X, y, classes = make_xy(data)
    n_classes = len(classes)

    X_tr, X_te, y_tr, y_te = train_test_split_np(
        X, y, test_size=0.2, random_state=random_state
    )
    X_tr, X_te = standardize(X_tr, X_te)

    print(f"Обучающая выборка: {len(X_tr)} объектов")
    print(f"Классы: {classes}")
    print("Распределение в train:",
          np.bincount(y_tr, minlength=n_classes).tolist())

    # --- CNN ---
    print("\nЗапуск CNN-отбора...")
    cnn_idx = cnn_select(X_tr, y_tr)
    print(f"  CNN сохранил {len(cnn_idx)} ({len(cnn_idx) / len(X_tr) * 100:.1f}%)")
    print("  по классам:",
          np.bincount(y_tr[cnn_idx], minlength=n_classes).tolist())

    # --- ENN ---
    print("\nЗапуск ENN-отбора...")
    enn_idx = enn_select(X_tr, y_tr, k=enn_k)
    print(f"  ENN сохранил {len(enn_idx)} ({len(enn_idx) / len(X_tr) * 100:.1f}%)")
    print("  по классам:",
          np.bincount(y_tr[enn_idx], minlength=n_classes).tolist())

    # --- метрики ---
    full_m = _eval_model(X_tr, y_tr, X_te, y_te, n_classes, k)
    cnn_m = _eval_model(X_tr[cnn_idx], y_tr[cnn_idx],
                        X_te, y_te, n_classes, k)
    enn_m = _eval_model(X_tr[enn_idx], y_tr[enn_idx],
                        X_te, y_te, n_classes, k)

    print("\n--- Метрики на тесте ---")
    for name, m in [("full", full_m), ("CNN", cnn_m), ("ENN", enn_m)]:
        print(f"  {name:>5s}  acc = {m['accuracy']:.4f}  "
              f"macro-F1 = {m['macro_f1']:.4f}  n_train = {m['n_train']}")

    # --- сохранение ---
    np.save(out_dir / "prototypes_cnn_idx.npy", cnn_idx)
    np.save(out_dir / "prototypes_enn_idx.npy", enn_idx)

    res = pd.DataFrame([
        {"variant": "full", **full_m},
        {"variant": "cnn", **cnn_m},
        {"variant": "enn", **enn_m},
    ])
    res.to_csv(out_dir / "selection_compare.csv", index=False)

    # --- график ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    variants = res["variant"].tolist()
    colors = ["C0", "C3", "C2"]
    for ax, (key, label) in zip(axes, [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("n_train", "Размер обучающей выборки"),
    ]):
        vals = res[key].tolist()
        bars = ax.bar(variants, vals, color=colors)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            txt = f"{int(v)}" if key == "n_train" else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, v, txt,
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"ParzenKNN (k={k}): full vs CNN vs ENN")
    fig.tight_layout()
    fig.savefig(out_dir / "selection_compare.png", dpi=150)
    plt.close(fig)

    print(f"\nСохранено: {out_dir / 'selection_compare.png'}, "
          f"{out_dir / 'selection_compare.csv'}, "
          f"{out_dir / 'prototypes_cnn_idx.npy'}, "
          f"{out_dir / 'prototypes_enn_idx.npy'}")
    return {"cnn": cnn_idx, "enn": enn_idx, "metrics": res}