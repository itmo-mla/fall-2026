import itertools
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kernels import is_positive_semidefinite, make_kernel
from plots import plot_grid, plot_margins, plot_models, plot_quality
from svm_dual import DualSVM

SEED = 42
OUT = Path(__file__).resolve().parent.parent


def prepare(X, y, test_size=0.3):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), 2.0 * y_tr - 1, 2.0 * y_te - 1


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def cv_grid(X, y, Cs, gammas, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    scores = np.zeros((len(Cs), len(gammas)))
    for (i, C), (j, g) in itertools.product(enumerate(Cs), enumerate(gammas)):
        accs = [
            np.mean(DualSVM(C=C, kernel="rbf", gamma=g).fit(X[a], y[a]).predict(X[b]) == y[b])
            for a, b in skf.split(X, y)
        ]
        scores[i, j] = np.mean(accs)
    return scores


def print_row(name, m):
    print(f"   {name:<14} acc {m['accuracy']:.4f}  prec {m['precision']:.4f}  rec {m['recall']:.4f}  f1 {m['f1']:.4f}")


def support_types(m):
    return f"периферийные {int((~m.support_).sum())}, граничные {int(m.boundary_.sum())}, нарушители {int(m.violators_.sum())}"


def compare(name, own, ref, X_te, y_te):
    pred_ref = np.where(ref.predict(X_te) > 0, 1.0, -1.0)
    m_own, m_ref = metrics(y_te, own.predict(X_te)), metrics(y_te, pred_ref)
    diff = np.abs(own.decision_function(X_te) - ref.decision_function(X_te)).max()
    same = np.mean(own.predict(X_te) == pred_ref)
    print(f"   [{name}] solver сошёлся: {own.converged_}, w0: {own.w0_:.5f} (sklearn: {-ref.intercept_[0]:.5f})")
    print_row("DualSVM", m_own)
    print_row("sklearn SVC", m_ref)
    print(f"   max|f_own - f_sklearn| = {diff:.4f}, совпадение предсказаний {same:.3f}, "
          f"опорных {int(own.support_.sum())} / {len(ref.support_)}")
    print(f"   типы объектов: {support_types(own)}")
    return m_own, m_ref


def main():
    print("=" * 70)
    print("Лабораторная работа №3: SVM (двойственная задача, трюк с ядром)")
    print("=" * 70)

    print("\n1. Датасет make_moons (двумерный, нелинейно разделимый)")
    Xm, ym = make_moons(n_samples=300, noise=0.25, random_state=SEED)
    X_tr, X_te, y_tr, y_te = prepare(Xm, ym)
    print(f"   Train: {len(y_tr)}, test: {len(y_te)}, классы -1/+1 в train: {int((y_tr < 0).sum())}/{int((y_tr > 0).sum())}")

    print("\n2. Подбор C и gamma для RBF-ядра (5-fold CV на train, собственный SVM)")
    Cs, gammas = [0.1, 1.0, 10.0, 100.0], [0.05, 0.2, 1.0, 5.0]
    scores = cv_grid(X_tr, y_tr, Cs, gammas)
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    C_best, g_best = Cs[i], gammas[j]
    print(f"   Лучшие параметры: C = {C_best}, gamma = {g_best} (CV accuracy {scores[i, j]:.4f})")
    plot_grid(scores, Cs, gammas, OUT / "svm_grid_cv.png", "Подбор C и gamma (RBF, make_moons)")

    print("\n3. Допустимость ядер: матрица Грама на train положительно полуопределена?")
    for name in ("linear", "poly", "rbf"):
        print(f"   {name:<7}: {is_positive_semidefinite(make_kernel(name, gamma=g_best)(X_tr, X_tr))}")

    print("\n4. Обучение и сравнение с sklearn.svm.SVC (тест)")
    configs = [
        ("linear", dict(kernel="linear"), dict(kernel="linear")),
        ("poly d=3", dict(kernel="poly", degree=3, coef0=1.0), dict(kernel="poly", gamma=1.0, degree=3, coef0=1.0)),
        ("rbf", dict(kernel="rbf", gamma=g_best), dict(kernel="rbf", gamma=g_best)),
    ]
    results, models = {}, []
    for name, own_kw, ref_kw in configs:
        own = DualSVM(C=C_best, **own_kw).fit(X_tr, y_tr)
        ref = SVC(C=C_best, **ref_kw).fit(X_tr, y_tr)
        m_own, m_ref = compare(name, own, ref, X_te, y_te)
        results[f"{name}: DualSVM"], results[f"{name}: sklearn"] = m_own, m_ref
        models.append((f"{name}, C={C_best:g}: acc на тесте {m_own['accuracy']:.3f}", own))
    w = models[0][1].w_
    print(f"   Линейный SVM: w = {np.round(w, 4)}, ширина полосы 2/||w|| = {2 / np.linalg.norm(w):.4f}")
    plot_models(models, X_tr, y_tr, OUT / "svm_boundaries.png",
                suptitle="Разделяющие поверхности (train): сплошная f=0, пунктир f=±1")
    plot_quality(results, OUT / "svm_quality.png")

    rbf = models[2][1]
    plot_margins(rbf, X_te, y_te, OUT / "rbf_margins.png", "Отступы тестовых объектов, RBF SVM")
    mg = rbf.margins(X_te, y_te)
    print(f"   RBF, отступы на тесте: min {mg.min():.3f}, mean {mg.mean():.3f}, ошибок {int((mg < 0).sum())}")

    print("\n5. Влияние C и gamma (RBF) на форму границы")
    sweep = [(f"RBF, gamma={g_best:g}, C={c:g}", DualSVM(C=c, kernel="rbf", gamma=g_best).fit(X_tr, y_tr))
             for c in (0.01, 1.0, 100.0)]
    sweep += [(f"RBF, C={C_best:g}, gamma={g:g}", DualSVM(C=C_best, kernel="rbf", gamma=g).fit(X_tr, y_tr))
              for g in (0.05, 1.0, 20.0)]
    for title, m in sweep:
        acc_tr, acc_te = np.mean(m.predict(X_tr) == y_tr), np.mean(m.predict(X_te) == y_te)
        print(f"   {title:<28} train {acc_tr:.3f}  test {acc_te:.3f}  опорных {int(m.support_.sum())}")
    plot_models(sweep, X_tr, y_tr, OUT / "svm_c_gamma.png", suptitle="Влияние C (верхний ряд) и gamma (нижний ряд)")

    print("\n6. Реальный датасет: Breast Cancer Wisconsin (бинарная классификация)")
    Xb, yb = load_breast_cancer(return_X_y=True)
    Xb_tr, Xb_te, yb_tr, yb_te = prepare(Xb, yb)
    gamma_b = 1.0 / Xb_tr.shape[1]
    print(f"   Train: {Xb_tr.shape}, test: {Xb_te.shape}, C = 1, gamma = 1/n_features = {gamma_b:.4f}")
    real = {}
    for name, kw in (("linear", dict(kernel="linear")), ("rbf", dict(kernel="rbf", gamma=gamma_b))):
        own = DualSVM(C=1.0, **kw).fit(Xb_tr, yb_tr)
        ref = SVC(C=1.0, **kw).fit(Xb_tr, yb_tr)
        real[f"{name}: DualSVM"], real[f"{name}: sklearn"] = compare(name, own, ref, Xb_te, yb_te)
    plot_quality(real, OUT / "svm_quality_breast_cancer.png")

    print("\nГрафики сохранены в директорию lab3.")


if __name__ == "__main__":
    main()
