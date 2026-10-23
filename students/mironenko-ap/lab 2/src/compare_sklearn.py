"""Сравнение нашей реализации с эталонным sklearn KNeighborsClassifier."""
from pathlib import Path
import numpy as np
import pandas as pd

from .data_utils import load_raw, make_xy, train_test_split_np, standardize
from .parzen_knn import ParzenKNN
from .metrics import accuracy, macro_f1, precision_recall_f1


def _full_metrics(y_true, y_pred, n_classes):
    prec, rec, f1 = precision_recall_f1(y_true, y_pred, n_classes)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "per_class_f1": f1.tolist(),
    }


def run_compare(out_dir: Path,
                grouping: str = "4",
                k_values=(1, 2, 3, 5, 7, 10, 15),
                random_state: int = 42):
    """
    Сравнивает три модели при разных k:
        - ParzenKNN (наш, окно переменной ширины, sample-point)
        - sklearn KNN (weights='uniform')
        - sklearn KNN (weights='distance')
    Сохраняет таблицу в report/figures/compare.csv
    """
    from sklearn.neighbors import KNeighborsClassifier

    df = load_raw()
    X, y, classes = make_xy(load_raw())
    n_classes = len(classes)

    X_tr, X_te, y_tr, y_te = train_test_split_np(
        X, y, test_size=0.2, random_state=random_state
    )
    X_tr, X_te = standardize(X_tr, X_te)

    rows = []
    for k in k_values:
        # --- наш ---
        our = ParzenKNN(k=k).fit(X_tr, y_tr)
        y_our = our.predict(X_te)
        m = _full_metrics(y_te, y_our, n_classes)
        rows.append({
            "model": "ParzenKNN (ours)", "k": k,
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
        })

        #  sklearn uniform / distance
        for w in ("uniform", "distance"):
            sk = KNeighborsClassifier(n_neighbors=k, weights=w)
            sk.fit(X_tr, y_tr)
            y_sk = sk.predict(X_te)
            m = _full_metrics(y_te, y_sk, n_classes)
            rows.append({
                "model": f"sklearn ({w})", "k": k,
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            })

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "compare.csv", index=False)

    print("\n Сравнение (test, macro-F1)")
    print(res.pivot(index="k", columns="model",
                    values="macro_f1").round(4).to_string())
    print(f"\nСохранено: {out_dir/'compare.csv'}")
    return res