import argparse
import os
import sys
import numpy as np

from src.data_loader import generate_dataset, save_dataset, load_dataset
from src.dual_svm import DualSVM
from src.reference import fit_reference, evaluate
from src.plots import plot_comparison, FIGURES_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORT_DIR = os.path.join(RESULTS_DIR, "report")


class Tee:
    """Дублирует поток: пишет одновременно в несколько потоков."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_args():
    p = argparse.ArgumentParser(
        description="Dual SVM: SMO vs scipy.optimize.minimize, RBF vs linear"
    )
    p.add_argument("--regenerate", action="store_true",
                   help="перегенерировать train/test CSV")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.03,
                   help="gamma для RBF (по умолчанию 0.03 ≈ 1/n_features)")
    p.add_argument("--kernel", type=str, default="rbf",
                   choices=["linear", "rbf", "poly"],
                   help="основное ядро; linear добавляется автоматически")
    p.add_argument("--no-plot", action="store_true",
                   help="не сохранять графики")
    return p.parse_args()


def _fit_and_report(svm, X_train, y_train, X_test, y_test, tag):
    svm.fit(X_train, y_train)
    acc, _ = evaluate(svm, X_test, y_test)
    print(f"  [{tag:<13}] acc={acc:.4f}  "
          f"SV={svm.n_support_vectors():3d}  "
          f"KKT={svm.max_kkt_violation_:.2e}  "
          f"iters={svm.n_iter_}")
    return svm, acc


def run(args):
    #  1. Данные
    if args.regenerate:
        data = generate_dataset()
        save_dataset(data)
    else:
        try:
            data = load_dataset()
        except FileNotFoundError:
            print("[main] датасет отсутствует — генерируем")
            data = generate_dataset()
            save_dataset(data)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_train_2d, X_test_2d = data["X_train_2d"], data["X_test_2d"]

    print(f"train: {X_train.shape}, test: {X_test.shape}  (30D)")
    print(f"PCA explained variance: {data['pca_ratio'].sum():.3f}")
    print(f"baseline (majority): "
          f"{max((y_test == 1).mean(), (y_test == -1).mean()):.4f}")

    #  2. Список ядер
    kernels = []
    for k in (args.kernel, "linear"):
        if k not in kernels:
            kernels.append(k)

    #  3. Прогон по ядрам
    for kernel in kernels:
        print(f"\n kernel={kernel}, C={args.C}, gamma={args.gamma}")

        # эталон
        ref = fit_reference(X_train, y_train,
                            C=args.C, gamma=args.gamma, kernel=kernel)
        acc_ref, _ = evaluate(ref, X_test, y_test)
        print(f"  [sklearn      ] acc={acc_ref:.4f}  "
              f"SV={ref.n_support_.sum():3d}")

        # custom smo
        svm_smo = DualSVM(C=args.C, kernel=kernel,
                          gamma=args.gamma, solver="smo")
        svm_smo, acc_smo = _fit_and_report(
            svm_smo, X_train, y_train, X_test, y_test, "custom smo"
        )

        # custom scipy
        svm_scipy = DualSVM(C=args.C, kernel=kernel,
                            gamma=args.gamma, solver="scipy")
        svm_scipy, acc_scipy = _fit_and_report(
            svm_scipy, X_train, y_train, X_test, y_test, "custom scipy"
        )

        print(f"  [diff smo-ref  ] {acc_smo - acc_ref:+.4f}")
        print(f"  [diff scipy-ref] {acc_scipy - acc_ref:+.4f}")
        print(f"  [diff smo-scipy] {acc_smo - acc_scipy:+.4f}")

        # восстановление w для линейного ядра
        if kernel == "linear":
            w = (svm_smo.lam * svm_smo.y_train) @ svm_smo.X_train
            w_ref = ref.coef_.ravel()
            cos_sim = float(
                w @ w_ref / (np.linalg.norm(w) * np.linalg.norm(w_ref))
            )
            print(f"  [linear] ||w_custom||={np.linalg.norm(w):.4f}  "
                  f"cos(w_custom, w_sklearn)={cos_sim:.6f}")

    #  4. Визуализация на PCA-2D
    if args.no_plot:
        print("\n[main] --no-plot: графики не сохраняем")
        return

    print("\n 2D-PCA visualisation")
    for kernel in kernels:
        svm_smo_2d = DualSVM(C=args.C, kernel=kernel,
                             gamma=args.gamma, solver="smo")
        svm_smo_2d.fit(X_train_2d, y_train)
        acc_s, _ = evaluate(svm_smo_2d, X_test_2d, y_test)

        svm_scipy_2d = DualSVM(C=args.C, kernel=kernel,
                               gamma=args.gamma, solver="scipy")
        svm_scipy_2d.fit(X_train_2d, y_train)
        acc_c, _ = evaluate(svm_scipy_2d, X_test_2d, y_test)

        ref_2d = fit_reference(X_train_2d, y_train,
                               C=args.C, gamma=args.gamma,
                               kernel=kernel)
        acc_r, _ = evaluate(ref_2d, X_test_2d, y_test)

        print(f"[{kernel}]  "
              f"smo={acc_s:.4f}  scipy={acc_c:.4f}  sklearn={acc_r:.4f}")

        plot_comparison(
            models=[svm_smo_2d, svm_scipy_2d, ref_2d],
            names=[f"Custom SMO ({kernel})",
                   f"Custom scipy ({kernel})",
                   f"sklearn SVC ({kernel})"],
            X=X_train_2d, y=y_train,
            title_suffix="  PCA 2D",
            fname=f"comparison_{kernel}.png",
        )


def main():
    args = parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    report_path = os.path.join(REPORT_DIR, "report.txt")

    orig_stdout = sys.stdout
    with open(report_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(orig_stdout, log_file)
        try:
            print(f"[main] отчёт: {report_path}")
            print(f"[main] аргументы: {vars(args)}")
            run(args)
            print(f"\n[main] готово. отчёт сохранён: {report_path}")
        finally:
            sys.stdout = orig_stdout


if __name__ == "__main__":
    main()