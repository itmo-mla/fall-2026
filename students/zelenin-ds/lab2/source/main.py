import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from dataset import *
from kernels import GaussianKernel
from metric_classifier import ParzenClassifier, PrototypeSelection
from utils import loo


K_N = 20
N_PROTO = 500
SEED = 43


def predict_batched(clf, X, ker, k, batch=200):
    """Предсказание пачками, чтобы не раздувать память тензором n×m×d."""
    return np.concatenate([clf.predict(X[i:i + batch], ker, k)
                           for i in range(0, len(X), batch)])


def gauss_weights(d):
    """Те же веса, что у ParzenClassifier: h = расстояние до k-го соседа."""
    h = np.maximum(d[:, -1:], 1e-12)
    return np.exp(-2 * (d / h) ** 2)


def ref_error(Xa, ya, Xb, yb, k):
    """Эталонная реализация из sklearn, аналогичная нашему Парзену."""
    m = KNeighborsClassifier(n_neighbors=k, weights=gauss_weights).fit(Xa, ya)
    return np.mean(m.predict(Xb) != yb)


if __name__ == "__main__":

    data = load_data("data/apple_quality.csv")
    data = prepare_data(data, drop_cols=DROP_COLS, target=TARGET)
    x = data.drop(columns=TARGET).to_numpy(dtype=float)
    y = data[TARGET].to_numpy(dtype=float).ravel()

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x))
    split = int(0.8 * len(x))
    tr, te = idx[:split], idx[split:]
    X_tr, X_te, y_tr, y_te = x[tr], x[te], y[tr], y[te]
    print("train:", X_tr.shape, "test:", X_te.shape)


    k_best, e_min, emp_risk = loo(X_tr, y_tr, k_n=K_N)
    ks = np.arange(1, len(emp_risk) + 1)
    print(f"Лучший k по LOO: {k_best}, ошибок {e_min}, риск {emp_risk[k_best - 1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(ks, emp_risk, label="Парзен (окно переменной ширины)")
    plt.axvline(k_best, color="gray", ls="--", label=f"k = {k_best}")
    plt.title("Зависимость LOO от числа соседей")
    plt.xlabel("Число соседей")
    plt.ylabel("Частота ошибок")
    plt.legend()
    plt.grid(alpha=0.3)


    ker = GaussianKernel()
    my_clf = ParzenClassifier()
    my_clf.fit(X_tr, y_tr)
    err_my = np.mean(predict_batched(my_clf, X_te, ker, k_best) != y_te)
    err_ref = ref_error(X_tr, y_tr, X_te, y_te, k_best)

    print("\n=== Сравнение с sklearn на тесте ===")
    print(f"Парзен (k={k_best}):          {err_my:.4f}")
    print(f"sklearn KNN (k={k_best}):     {err_ref:.4f}")


    Xs, ys = X_tr[:N_PROTO], y_tr[:N_PROTO]
    ps_k = k_best if k_best % 2 == 1 else k_best + 1
    ps = PrototypeSelection(k=ps_k)
    ps.fit(Xs, ys)
    alive = ps.select()
    n_proto = int(alive.sum())
    print(f"\nЭталонов: {n_proto} из {N_PROTO} (k отбора = {ps_k})")


    Xc = Xs - Xs.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for cls, color in [(-1, "tab:blue"), (1, "tab:red")]:
        rem = (ys == cls) & ~alive
        kept = (ys == cls) & alive
        ax[0].scatter(Z[rem, 0], Z[rem, 1], c=color, alpha=0.2, s=25,
                      label=f"класс {int(cls)}: удалены")
        ax[0].scatter(Z[kept, 0], Z[kept, 1], c=color, s=90, edgecolors="black",
                      label=f"класс {int(cls)}: эталоны")
    ax[0].set_title("Эталоны в проекции на 2 главные компоненты")
    ax[0].legend(fontsize=8)

    sizes = [N_PROTO] + [h[0] for h in ps.history]
    errs = [ps.error] + [h[1] for h in ps.history]
    ax[1].plot(sizes, errs)
    ax[1].invert_xaxis()
    ax[1].set_xlabel("Размер набора эталонов")
    ax[1].set_ylabel("Ошибка на объектах подвыборки")
    ax[1].set_title("Ход отбора")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()


    ks_cmp = [k for k in [1, 3, 5, 7, 9, 11] if k <= n_proto]
    err_all = [ref_error(Xs, ys, X_te, y_te, k) for k in ks_cmp]
    err_pro = [ref_error(Xs[alive], ys[alive], X_te, y_te, k) for k in ks_cmp]

    print("\n=== Ошибка на тесте: KNN с отбором и без ===")
    print(f"{'k':>3} | {'без отбора (' + str(N_PROTO) + ')':>18} | {'с отбором (' + str(n_proto) + ')':>18}")
    for k, a, p in zip(ks_cmp, err_all, err_pro):
        print(f"{k:>3} | {a:>18.4f} | {p:>18.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(ks_cmp, err_all, marker="o", label=f"без отбора ({N_PROTO})")
    plt.plot(ks_cmp, err_pro, marker="o", label=f"с отбором ({n_proto})")
    plt.xlabel("k")
    plt.ylabel("Ошибка на тесте")
    plt.title("KNN с отбором эталонов и без")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()