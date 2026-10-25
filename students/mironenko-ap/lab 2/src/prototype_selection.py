"""Отбор эталонов: CNN  и ENN."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import load_raw, make_xy, train_test_split_np, standardize
from .parzen_knn import ParzenKNN
from .metrics import accuracy, macro_f1


# 1. CNN — Condensed Nearest Neighbor (для 1-NN)
def cnn_select(X, y):
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


# 2. ENN — Edited Nearest Neighbor (для любого k)
def enn_select(X, y, k: int = 3):
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

# 2b. CCV — Compactness Criterion (профиль компактности)
def _self_excluded_knn(X, k: int):
    n = len(X)
    diffs = X[:, None, :] - X[None, :, :]
    sq = np.einsum("ijk,ijk->ij", diffs, diffs)
    np.fill_diagonal(sq, np.inf)
    k_eff = min(k, n - 1)
    return np.argpartition(sq, k_eff - 1, axis=1)[:, :k_eff]


def ccv_profiles(X, y, k: int = 1):
    n = len(X)
    knn_idx = _self_excluded_knn(X, k)
    Pi = np.zeros(n, dtype=int)
    R = np.zeros(n, dtype=int)
    for j in range(n):
        for m in knn_idx[j]:
            if y[m] == y[j]:
                Pi[m] += 1
            else:
                R[m] += 1
    return Pi, R, Pi + R


def ccv_select(X, y,
               k: int = 1,
               variant: str = "pi_gt_r",
               threshold: float = 0.0):
    Pi, R, Sigma = ccv_profiles(X, y, k=k)

    if variant == "pi_minus_r":
        score = Pi - R
        sel = np.where(score > threshold)[0]
    elif variant == "pi_gt_r":
        sel = np.where(Pi > R)[0]
    elif variant == "pi1":
        sel = np.where(Pi > 0)[0]
    elif variant == "sigma_pi_r":
        score = (Sigma * Pi) / np.maximum(R, 1)
        sel = np.where(score > threshold)[0]
    elif variant == "pi_over_r":
        score = Pi / np.maximum(R, 1)
        sel = np.where(score > threshold)[0]
    else:
        raise ValueError(f"Unknown CCV variant: {variant!r}")

    return sel

# 3. Оценка
def _eval_model(X_tr, y_tr, X_te, y_te, n_classes, k):
    model = ParzenKNN(k=k).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "accuracy": accuracy(y_te, y_pred),
        "macro_f1": macro_f1(y_te, y_pred, n_classes),
        "n_train": len(X_tr),
    }



# 4. Основной сценарий: сравнение full / CNN / ENN / CCV
def run_selection(out_dir: Path,
                  k: int = 3,
                  enn_k: int = 3,
                  ccv_k: int = 1,
                  ccv_variant: str = "pi_gt_r",
                  random_state: int = 42):
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

    #  CNN
    print("\nЗапуск CNN-отбора...")
    cnn_idx = cnn_select(X_tr, y_tr)
    print(f"  CNN сохранил {len(cnn_idx)} ({len(cnn_idx)/len(X_tr)*100:.1f}%)")

    #  ENN
    print("\nЗапуск ENN-отбора...")
    enn_idx = enn_select(X_tr, y_tr, k=enn_k)
    print(f"  ENN сохранил {len(enn_idx)} ({len(enn_idx)/len(X_tr)*100:.1f}%)")

    #  CCV
    print(f"\nЗапуск CCV-отбора (k={ccv_k}, variant={ccv_variant})...")
    ccv_idx = ccv_select(X_tr, y_tr, k=ccv_k, variant=ccv_variant)
    print(f"  CCV сохранил {len(ccv_idx)} ({len(ccv_idx)/len(X_tr)*100:.1f}%)")
    print("  по классам:",
          np.bincount(y_tr[ccv_idx], minlength=n_classes).tolist())
    # профили
    Pi, R, Sigma = ccv_profiles(X_tr, y_tr, k=ccv_k)
    print(f"  Π(1): mean={Pi.mean():.2f}, max={Pi.max()}")
    print(f"  R:    mean={R.mean():.2f}, max={R.max()}")
    print(f"  отобрано с R=0: {(R[ccv_idx] == 0).sum()}/{len(ccv_idx)}")
    print(f"  отобрано с Π>0: {(Pi[ccv_idx] > 0).sum()}/{len(ccv_idx)}")

    #  метрики
    full_m = _eval_model(X_tr, y_tr, X_te, y_te, n_classes, k)
    cnn_m  = _eval_model(X_tr[cnn_idx], y_tr[cnn_idx],
                         X_te, y_te, n_classes, k)
    enn_m  = _eval_model(X_tr[enn_idx], y_tr[enn_idx],
                         X_te, y_te, n_classes, k)
    ccv_m  = _eval_model(X_tr[ccv_idx], y_tr[ccv_idx],
                         X_te, y_te, n_classes, k)

    print("\n Метрики на тесте ")
    for name, m in [("full", full_m), ("CNN", cnn_m),
                    ("ENN", enn_m), ("CCV", ccv_m)]:
        print(f"  {name:>4s}  acc = {m['accuracy']:.4f}  "
              f"macro-F1 = {m['macro_f1']:.4f}  n_train = {m['n_train']}")

    #  сохранение
    np.save(out_dir / "prototypes_cnn_idx.npy", cnn_idx)
    np.save(out_dir / "prototypes_enn_idx.npy", enn_idx)
    np.save(out_dir / "prototypes_ccv_idx.npy", ccv_idx)       # НОВОЕ
    np.save(out_dir / "ccv_profiles_Pi.npy",  Pi)              # НОВОЕ
    np.save(out_dir / "ccv_profiles_R.npy",   R)               # НОВОЕ

    res = pd.DataFrame([
        {"variant": "full", **full_m},
        {"variant": "cnn",  **cnn_m},
        {"variant": "enn",  **enn_m},
        {"variant": "ccv",  **ccv_m},                          # НОВОЕ
    ])
    res.to_csv(out_dir / "selection_compare.csv", index=False)

    #  график
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    variants = res["variant"].tolist()
    colors = ["C0", "C3", "C2", "C4"]                          # +CCV
    for ax, (key, label) in zip(axes, [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("n_train",  "Размер обучающей выборки"),
    ]):
        vals = res[key].tolist()
        bars = ax.bar(variants, vals, color=colors)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            txt = f"{int(v)}" if key == "n_train" else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width()/2, v, txt,
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"ParzenKNN (k={k}): full vs CNN vs ENN vs CCV")
    fig.tight_layout()
    fig.savefig(out_dir / "selection_compare.png", dpi=150)
    plt.close(fig)

    print(f"\nСохранено: {out_dir/'selection_compare.png'}, "
          f"{out_dir/'selection_compare.csv'}, "
          f"{out_dir/'prototypes_ccv_idx.npy'}")
    return {"cnn": cnn_idx, "enn": enn_idx, "ccv": ccv_idx,
            "metrics": res}