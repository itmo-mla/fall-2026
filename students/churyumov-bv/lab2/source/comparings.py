import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

import knn
import selecting


def _sklearn_predict(D_fit, y_fit, D_pred, k):
    """KNN из sklearn на готовых расстояниях (те же, что у нашего алгоритма)"""
    model = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    return model.fit(D_fit, y_fit).predict(D_pred)


def compare_with_prototypes(D_train, D_test, y_train, y_test, ks, best_k, k_ctrl):
    """
    Сравнение KNN с отбором эталонов и без отбора.

    D_train  - расстояния train × train
    D_test   - расстояния test × train
    ks       - диапазон k для подбора по LOO на множестве эталонов
    best_k   - k, подобранный по LOO на полной обучающей выборке
    k_ctrl   - длина контроля CCV при отборе эталонов

    Возвращает словарь: omega (индексы эталонов), history (CCV по шагам), best_k_omega.
    """
    Y = np.unique(y_train)
    K = knn.gaussian_kernel

    # 6. отбор эталонов
    omega, history = selecting.select_prototypes(D_train, y_train, k_ctrl)
    print(f"Отбор эталонов: |Ω| = {len(omega)} из {len(y_train)} "
          f"({100 * len(omega) / len(y_train):.1f}%), "
          f"CCV {history[0]:.4f} -> {history[-1]:.4f}")

    # k заново подбираем по LOO уже на множестве эталонов
    D_omega = D_train[np.ix_(omega, omega)]
    y_omega = y_train[omega]
    ks_omega = [k for k in ks if k <= len(omega) - 2]
    risks_omega = knn.loo_curve(D_omega, y_omega, Y, ks_omega, K)
    best_k_omega = ks_omega[int(np.argmin(risks_omega))]
    print(f"k по LOO: без отбора {best_k}, с отбором {best_k_omega}")

    # 8. качество на тесте: без отбора (вся выборка) и с отбором (только эталоны)
    D_test_omega = D_test[:, omega]
    predictions = {
        "Parzen-KNN": (
            knn.predict(D_test, y_train, Y, best_k, K),
            knn.predict(D_test_omega, y_omega, Y, best_k_omega, K),
        ),
        "sklearn KNN": (
            _sklearn_predict(D_train, y_train, D_test, best_k),
            _sklearn_predict(D_omega, y_omega, D_test_omega, best_k_omega),
        ),
        "1NN": (
            _sklearn_predict(D_train, y_train, D_test, 1),
            _sklearn_predict(D_omega, y_omega, D_test_omega, 1),
        )
    }

    print(f"{'метод':18s} | {'без отбора':^20s} | {'с отбором':^20s}")
    print(f"{'':18s} | {'accuracy':>9s} {'F1':>9s}  | {'accuracy':>9s} {'F1':>9s}")
    for name, (y_full, y_proto) in predictions.items():
        print(f"{name:18s} | {accuracy_score(y_test, y_full):9.4f} {f1_score(y_test, y_full):9.4f}  | "
              f"{accuracy_score(y_test, y_proto):9.4f} {f1_score(y_test, y_proto):9.4f}")

    return {"omega": omega, "history": history, "best_k_omega": best_k_omega}
