"""Шаг 6: сравнение собственной реализации с sklearn.svm.SVC."""
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classifier import DualSVM


def print_metrics(y_test, pred):
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall: {recall_score(y_test, pred):.4f}")
    print(f"F1: {f1_score(y_test, pred):.4f}")


def fit_own(X_train, y_train, X_test, y_test, name, **kw):
    t0 = time.time()
    model = DualSVM(**kw).fit(X_train, y_train)
    t = time.time() - t0
    pred = model.predict(X_test)
    print(f"[{name}] Обучение: {t:.3f} c, опорных векторов: {model.n_sv_} / {len(X_train)}")
    print_metrics(y_test, pred)
    return model, pred, t


def fit_sklearn(X_train, y_train, X_test, y_test, name, **kw):
    t0 = time.time()
    model = SVC(**kw).fit(X_train, y_train)
    t = time.time() - t0
    pred = model.predict(X_test)
    print(f"sklearn, {name}:")
    print(f"  Обучение: {t:.3f} c, SV: {model.n_support_.sum()}, "
          f"Accuracy: {accuracy_score(y_test, pred):.4f}, F1: {f1_score(y_test, pred):.4f}")
    return model, pred, t


def results_table(y_test, runs):
    """runs: список (имя, предсказания, число SV, время)."""
    return pd.DataFrame({
        'model': [r[0] for r in runs],
        'accuracy': [accuracy_score(y_test, r[1]) for r in runs],
        'precision': [precision_score(y_test, r[1]) for r in runs],
        'recall': [recall_score(y_test, r[1]) for r in runs],
        'f1': [f1_score(y_test, r[1]) for r in runs],
        'n_support_vectors': [r[2] for r in runs],
        'train_time_s': [r[3] for r in runs],
    })


def print_agreement(pred_own_lin, pred_sk_lin, pred_own_rbf, pred_sk_rbf):
    print(f"Линейное ядро: совпадение предсказаний {(pred_own_lin == pred_sk_lin).mean():.4f}")
    print(f"RBF-ядро: совпадение предсказаний {(pred_own_rbf == pred_sk_rbf).mean():.4f}")
