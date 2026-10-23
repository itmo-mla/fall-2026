"""Влияние дисбаланса: подвыборка 1:2 против полной выборки (3.4% отказов).

Вероятностная постановка (сл. 20-21): модель оценивает P(y=+1|x). При смене априорной доли
класса меняется в первую очередь свободный член: ln(odds_full) - ln(odds_sub).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

from common import setup_style, save_fig, C1, C2, INK2
from logreg import LogisticRegressionIRLS
from lr5_compare import sklearn_reference
from lr5_data import get_subsample_split, get_full_split, FEATURE_NAMES, save_csv

COEF_NAMES = ["intercept"] + FEATURE_NAMES


def best_f1_threshold(y01, p):
    """Порог, максимизирующий F1 на train."""
    pr, rc, th = precision_recall_curve(y01, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[np.argmax(f1)])


def metrics(model, X_tr, y_tr, X_te, y_te):
    y01_tr, y01 = (y_tr == 1).astype(int), (y_te == 1).astype(int)
    p_tr, p = model.predict_proba(X_tr), model.predict_proba(X_te)
    thr = best_f1_threshold(y01_tr, p_tr)
    out = {
        "доля +1 в test": y01.mean(),
        "средняя P(+1|x), test": p.mean(),
        "log-loss": log_loss(y01, p),
        "Brier": brier_score_loss(y01, p),
        "ROC-AUC": roc_auc_score(y01, p),
        "PR-AUC (AP)": average_precision_score(y01, p),
        "accuracy @0.5": accuracy_score(y01, p >= 0.5),
        "accuracy 'всегда -1'": 1 - y01.mean(),
        "precision @0.5": precision_score(y01, p >= 0.5, zero_division=0),
        "recall @0.5": recall_score(y01, p >= 0.5),
        "порог F1 (train)": thr,
        "precision @F1-порог": precision_score(y01, p >= thr, zero_division=0),
        "recall @F1-порог": recall_score(y01, p >= thr),
        "F1 @F1-порог": f1_score(y01, p >= thr),
    }
    return out, p


def main():
    setup_style()
    sets = {"Подвыборка 1:2": get_subsample_split(), "Полная выборка": get_full_split()}
    fits, rows, probs = {}, {}, {}
    for name, (X_tr, X_te, y_tr, y_te) in sets.items():
        m = LogisticRegressionIRLS("newton").fit(X_tr, y_tr)
        rows[name], probs[name] = metrics(m, X_tr, y_tr, X_te, y_te)
        fits[name] = (m, y_tr, y_te)
        # контроль: те же метрики у sklearn
        sk = sklearn_reference(X_tr, y_tr)
        y01 = (y_te == 1).astype(int)
        d = abs(roc_auc_score(y01, sk.predict_proba(X_te)[:, 1]) - rows[name]["ROC-AUC"])
        print(f"[{name}] |ROC-AUC_own - ROC-AUC_sklearn| = {d:.1e}")

    keys = list(next(iter(rows.values())).keys())
    print(f"\n{'метрика (test)':<26}{'Подвыборка 1:2':>16}{'Полная выборка':>16}")
    for k in keys:
        a, b = rows["Подвыборка 1:2"][k], rows["Полная выборка"][k]
        print(f"{k:<26}{a:>16.4f}{b:>16.4f}")

    # --- размеры test ---
    for name, (_, _, y_te) in fits.items():
        print(f"test [{name}]: n={len(y_te)}, положительных={(y_te == 1).sum()}")

    # --- смещение свободного члена (сл. 20-21: ln odds) ---
    def log_odds(y):
        pos = (y == 1).sum()
        return np.log(pos / (len(y) - pos))
    ms, ys, _ = fits["Подвыборка 1:2"]
    mf, yf, _ = fits["Полная выборка"]
    shift = log_odds(yf) - log_odds(ys)
    print(f"\nln odds: подвыборка (train) = {log_odds(ys):.4f}, полная (train) = {log_odds(yf):.4f},"
          f" разность = {shift:.4f}")
    print(f"b_full - b_sub = {mf.intercept_ - ms.intercept_:.4f} (по теории при точной модели: "
          f"{shift:.4f})")
    print(f"\n{'вес':<14}{'подвыборка':>12}{'полная':>10}{'полная-подвыб.':>16}")
    ws, wf = ms.res_.w, mf.res_.w
    for nm, a, b in zip(COEF_NAMES, ws, wf):
        print(f"{nm:<14}{a:>12.4f}{b:>10.4f}{b - a:>16.4f}")

    # --- CSV ---
    en = {"доля +1 в test": "test_positive_rate", "средняя P(+1|x), test": "test_mean_probability",
          "log-loss": "log_loss", "Brier": "brier", "ROC-AUC": "roc_auc", "PR-AUC (AP)": "pr_auc_ap",
          "accuracy @0.5": "accuracy_at_0.5", "accuracy 'всегда -1'": "accuracy_always_minus1",
          "precision @0.5": "precision_at_0.5", "recall @0.5": "recall_at_0.5",
          "порог F1 (train)": "f1_threshold_train", "precision @F1-порог": "precision_at_f1_threshold",
          "recall @F1-порог": "recall_at_f1_threshold", "F1 @F1-порог": "f1_at_f1_threshold"}
    save_csv(pd.DataFrame({
        "metric": [en[k] for k in keys],
        "subsample_1_2": [rows["Подвыборка 1:2"][k] for k in keys],
        "full": [rows["Полная выборка"][k] for k in keys]}), "lr5_metrics.csv")
    ids = {"Подвыборка 1:2": "subsample_1_2", "Полная выборка": "full"}
    save_csv(pd.concat([pd.DataFrame({"dataset": ids[n], "y_true": sets[n][3],
                                      "p_newton": probs[n]}) for n in sets], ignore_index=True),
             "lr5_test_predictions.csv")

    # --- график: ROC и PR ---
    cols = {"Подвыборка 1:2": C1, "Полная выборка": C2}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, (m, ytr, y_te) in fits.items():
        y01, p = (y_te == 1).astype(int), probs[name]
        fpr, tpr, _ = roc_curve(y01, p)
        a1.plot(fpr, tpr, color=cols[name],
                label=f"{name}, AUC = {rows[name]['ROC-AUC']:.3f}")
        pr, rc, _ = precision_recall_curve(y01, p)
        a2.plot(rc, pr, color=cols[name],
                label=f"{name}, AP = {rows[name]['PR-AUC (AP)']:.3f}")
        a2.axhline(y01.mean(), color=cols[name], ls=":", lw=1.4)
    a1.plot([0, 1], [0, 1], "--", color=INK2, lw=1)
    a1.set_xlabel("FPR"); a1.set_ylabel("TPR"); a1.set_title("ROC-кривая (test)")
    a1.legend(loc="lower right")
    a2.set_xlabel("recall"); a2.set_ylabel("precision")
    a2.set_title("PR-кривая (test); пунктир — доля класса +1")
    a2.legend(loc="lower left")
    save_fig(fig, "lr5_imbalance.png")


if __name__ == "__main__":
    main()
