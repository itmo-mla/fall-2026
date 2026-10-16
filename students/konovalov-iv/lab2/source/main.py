import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from dataset import load_data
from knn import ParzenKNN, gaussian_kernel, loo_risks, one_nn_predict
from metrics import accuracy, macro_f1
from plots import plot_ccv_paths, plot_compactness, plot_loo, plot_prototypes
from prototypes import CCVSelector

PLOTS = Path(__file__).resolve().parent / "plots"
K_MAX = 40


def sklearn_parzen_weights(dist):
    h = np.maximum(dist[:, -1:], 1e-12)
    w = gaussian_kernel(dist / h)
    w[:, -1] = 0
    return w


def predict_time(predict, X, repeats=100):
    start = time.perf_counter()
    for _ in range(repeats):
        predict(X)
    return (time.perf_counter() - start) / repeats * 1000


def main():
    PLOTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, class_names = load_data()
    n_classes = len(class_names)
    print(f"train: {X_train.shape}, test: {X_test.shape}, классы train: {np.bincount(y_train)}")

    ks = np.arange(1, K_MAX + 1)
    risks = loo_risks(X_train, y_train, ks)
    risks_biased = loo_risks(X_train, y_train, ks, biased=True)
    k_best = int(ks[np.argmin(risks)])
    print(f"LOO: k* = {k_best}, риск {risks.min():.4f}")
    print("LOO(k):", " ".join(f"{k}:{r:.3f}" for k, r in zip(ks, risks)))


    grid = GridSearchCV(KNeighborsClassifier(weights=sklearn_parzen_weights),
                        {"n_neighbors": list(ks + 1)}, cv=LeaveOneOut()).fit(X_train, y_train)
    sk_risks = 1 - grid.cv_results_["mean_test_score"]
    print(f"LOO-кривые своей реализации и sklearn совпадают: {np.allclose(risks, sk_risks)}")
    sk_grid = GridSearchCV(KNeighborsClassifier(),
                           {"n_neighbors": list(ks), "weights": ["uniform", "distance"]},
                           cv=LeaveOneOut()).fit(X_train, y_train)
    print(f"sklearn KNN: лучшие параметры {sk_grid.best_params_}, LOO-риск {1 - sk_grid.best_score_:.4f}")

    own = ParzenKNN(k=k_best).fit(X_train, y_train)
    sk_parzen = KNeighborsClassifier(n_neighbors=k_best + 1, weights=sklearn_parzen_weights).fit(X_train, y_train)
    sk_best = sk_grid.best_estimator_
    print("предсказания своей реализации и sklearn с тем же окном совпадают:",
          np.array_equal(own.predict(X_test), sk_parzen.predict(X_test)))

    rows = {
        f"ParzenKNN, k={k_best} (своя)": (own.predict, len(y_train)),
        f"sklearn, окно Парзена, k={k_best}": (sk_parzen.predict, len(y_train)),
        f"sklearn KNN, k={sk_grid.best_params_['n_neighbors']}, {sk_grid.best_params_['weights']}":
            (sk_best.predict, len(y_train)),
        "1NN, вся выборка": (lambda X: one_nn_predict(X, X_train, y_train), len(y_train)),
    }

    selector = CCVSelector(X_train, y_train, control_fraction=0.3)
    full = np.ones(len(y_train), dtype=bool)
    print(f"\nCCV на всей выборке: {selector.ccv(full):.4f}")

    start = time.perf_counter()
    omega_remove, remove_path, n_removed = selector.greedy_remove()
    print(f"удаление: {omega_remove.sum()} эталонов, CCV {selector.ccv(omega_remove):.4f}, "
          f"{time.perf_counter() - start:.1f} с")
    start = time.perf_counter()
    omega_add, add_path = selector.greedy_add(X_train)
    print(f"добавление: {omega_add.sum()} эталонов, CCV {selector.ccv(omega_add):.4f}, "
          f"{time.perf_counter() - start:.1f} с")

    for name, omega in [("удаление", omega_remove), ("добавление", omega_add)]:
        X_ref, y_ref = X_train[omega], y_train[omega]
        ks_omega = np.arange(1, min(K_MAX, omega.sum() - 2) + 1)
        k_omega = int(ks_omega[np.argmin(loo_risks(X_train, y_train, ks_omega, ref_mask=omega))])
        model = ParzenKNN(k=k_omega).fit(X_ref, y_ref)
        rows[f"1NN, эталоны ({name})"] = (lambda X, a=X_ref, b=y_ref: one_nn_predict(X, a, b), omega.sum())
        rows[f"ParzenKNN, k={k_omega}, эталоны ({name})"] = (model.predict, omega.sum())

    print("\n| Модель | Accuracy | Macro-F1 | Опорных объектов | Сжатие | Предсказание, мс |")
    print("|---|---|---|---|---|---|")
    for name, (predict, n_ref) in rows.items():
        y_pred = predict(X_test)
        print(f"| {name} | {accuracy(y_test, y_pred):.3f} | {macro_f1(y_test, y_pred):.3f} | {n_ref} "
              f"| {1 - n_ref / len(y_train):.0%} | {predict_time(predict, X_test):.3f} |")

    plot_loo(ks, risks, risks_biased, k_best, PLOTS / "loo.png")
    m_max = min(20, omega_remove.sum() - 1)
    plot_compactness({
        "вся обучающая выборка": selector.compactness_profile(full, m_max),
        "эталоны (удаление)": selector.compactness_profile(omega_remove, m_max),
        "эталоны (добавление)": selector.compactness_profile(omega_add, m_max),
    }, PLOTS / "compactness.png")
    plot_ccv_paths(remove_path, n_removed, add_path, n_classes, PLOTS / "ccv.png")
    plot_prototypes(X_train, y_train, {"Жадное удаление": omega_remove, "Жадное добавление": omega_add},
                    class_names, PLOTS / "prototypes.png")
    print(f"\nграфики сохранены в {PLOTS}")


if __name__ == "__main__":
    main()
