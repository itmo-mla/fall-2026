import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from dataset import load_cancer, load_moons
from plots import plot_C, plot_cancer_C, plot_kernels, plot_sv_types
from svm import KernelSVM, linear_kernel, poly_kernel, rbf_kernel

PLOTS = Path(__file__).resolve().parent / "plots"


def timed(f):
    start = time.perf_counter()
    result = f()
    return result, time.perf_counter() - start


def kernel_configs(gamma):
    return [
        ("линейное", linear_kernel(), dict(kernel="linear")),
        ("полиномиальное d=3", poly_kernel(3), dict(kernel="poly", degree=3, gamma=1.0, coef0=1.0)),
        ("RBF", rbf_kernel(gamma), dict(kernel="rbf", gamma=gamma)),
    ]


def main():
    PLOTS.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_moons()
    print(f"moons: train {X_train.shape}, test {X_test.shape}")
    gamma = 1.0
    rows = []
    print("\nmoons, C = 1")
    for name, kern, ref_kw in kernel_configs(gamma):
        own, t_own = timed(lambda: KernelSVM(kern, C=1.0).fit(X_train, y_train))
        ref, t_ref = timed(lambda: SVC(C=1.0, tol=1e-6, **ref_kw).fit(X_train, y_train))
        po, pr = own.predict(X_test), ref.predict(X_test)
        print(f"{name:20s} own acc {accuracy_score(y_test, po):.3f} sk acc {accuracy_score(y_test, pr):.3f} "
              f"| SV {len(own.sv_index_)}/{len(ref.support_)} | совпадение предсказаний {np.mean(po == pr):.3f} "
              f"| t {t_own:.2f}/{t_ref:.3f} c | SLSQP ok={own.opt_result_.success}")
        rows.append((name, own, ref))
    plot_kernels(rows, X_train, y_train, PLOTS / "kernels.png")

    models = [(C, KernelSVM(rbf_kernel(gamma), C=C).fit(X_train, y_train)) for C in [0.01, 1, 100]]
    plot_C(models, X_train, y_train, PLOTS / "C_effect.png")
    for C, m in models:
        print(f"RBF C={C}: test acc {accuracy_score(y_test, m.predict(X_test)):.3f}, SV {len(m.sv_index_)}")
    plot_sv_types(rows[2][1], X_train, y_train, PLOTS / "sv_types.png")
    counts = np.bincount(rows[2][1].sv_types(), minlength=3)
    print("типы объектов (RBF, C=1): периферийные %d, граничные %d, нарушители %d" % tuple(counts))

    X_train, X_test, y_train, y_test = load_cancer()
    print(f"\ncancer: train {X_train.shape}, test {X_test.shape}")
    g = 1.0 / X_train.shape[1]
    for name, kern, kw in kernel_configs(g):
        own, t_own = timed(lambda: KernelSVM(kern, C=1.0).fit(X_train, y_train))
        ref, t_ref = timed(lambda: SVC(C=1.0, tol=1e-6, **kw).fit(X_train, y_train))
        po, pr = own.predict(X_test), ref.predict(X_test)
        print(f"{name:20s} own acc {accuracy_score(y_test, po):.3f} F1 {f1_score(y_test, po):.3f} | "
              f"sk acc {accuracy_score(y_test, pr):.3f} F1 {f1_score(y_test, pr):.3f} | "
              f"SV {len(own.sv_index_)}/{len(ref.support_)} | совпадение {np.mean(po == pr):.3f} "
              f"| t {t_own:.2f}/{t_ref:.3f} c")

    Cs = [0.01, 0.1, 1, 10, 100, 1000]
    own_acc, ref_acc, n_sv = [], [], []
    for C in Cs:
        own = KernelSVM(rbf_kernel(g), C=C).fit(X_train, y_train)
        ref = SVC(C=C, kernel="rbf", gamma=g, tol=1e-6).fit(X_train, y_train)
        own_acc.append(accuracy_score(y_test, own.predict(X_test)))
        ref_acc.append(accuracy_score(y_test, ref.predict(X_test)))
        n_sv.append(len(own.sv_index_))
        print(f"RBF C={C:<7} own acc {own_acc[-1]:.3f} sk acc {ref_acc[-1]:.3f} SV {n_sv[-1]}/{len(ref.support_)}")
    plot_cancer_C(Cs, own_acc, ref_acc, n_sv, PLOTS / "cancer_C.png")


if __name__ == "__main__":
    main()
