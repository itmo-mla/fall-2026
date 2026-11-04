import argparse
import os
import sys

from src.data_loader import generate_dataset, save_dataset, load_dataset
from src.dual_svm import DualSVM
from src.reference import fit_reference, evaluate

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
        description="Диагностическая сетка по gamma для DualSVM vs sklearn"
    )
    p.add_argument("--regenerate", action="store_true",
                   help="перегенерировать train/test CSV")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--kernel", type=str, default="rbf",
                   choices=["linear", "rbf", "poly"])
    p.add_argument("--gammas", type=float, nargs="+",
                   default=[0.005, 0.01, 0.03, 0.1, 0.3, 1.0],
                   help="список значений gamma")
    return p.parse_args()


def run(args):
    #  данные
    if args.regenerate:
        data = generate_dataset()
        save_dataset(data)
    else:
        try:
            data = load_dataset()
        except FileNotFoundError:
            print("[check_gamma] датасет отсутствует — генерируем")
            data = generate_dataset()
            save_dataset(data)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"PCA explained variance: {data['pca_ratio'].sum():.3f}")
    print(f"kernel={args.kernel}, C={args.C}\n")

    header = (f"{'gamma':>8} | "
              f"{'smo acc':>8} {'SV':>4} | "
              f"{'scipy acc':>9} {'SV':>4} | "
              f"{'sklearn acc':>11} {'SV':>4} | "
              f"{'diff(smo-ref)':>13}")
    print(header)
    print("-" * len(header))

    rows = []
    for gamma in args.gammas:
        # custom SMO
        svm_smo = DualSVM(C=args.C, kernel=args.kernel,
                          gamma=gamma, solver="smo")
        svm_smo.fit(X_train, y_train)
        acc_smo, _ = evaluate(svm_smo, X_test, y_test)

        # custom scipy
        svm_scipy = DualSVM(C=args.C, kernel=args.kernel,
                            gamma=gamma, solver="scipy")
        svm_scipy.fit(X_train, y_train)
        acc_scipy, _ = evaluate(svm_scipy, X_test, y_test)

        # sklearn
        ref = fit_reference(X_train, y_train,
                            C=args.C, gamma=gamma, kernel=args.kernel)
        acc_ref, _ = evaluate(ref, X_test, y_test)

        diff = acc_smo - acc_ref
        rows.append((gamma, acc_smo, acc_scipy, acc_ref, diff))

        print(f"{gamma:8.3f} | "
              f"{acc_smo:8.4f} {svm_smo.n_support_vectors():4d} | "
              f"{acc_scipy:9.4f} {svm_scipy.n_support_vectors():4d} | "
              f"{acc_ref:11.4f} {ref.n_support_.sum():4d} | "
              f"{diff:+13.4f}")

    #  итог
    best_gamma, best_smo, _, best_ref, _ = max(rows, key=lambda r: r[3])
    print(f"\nЛучшее (по sklearn): gamma={best_gamma:.3f} "
          f"(sklearn acc={best_ref:.4f}, smo acc={best_smo:.4f})")

    print("\nИтог по расхождениям:")
    for gamma, acc_smo, acc_scipy, acc_ref, diff in rows:
        print(f"  gamma={gamma:<6.3f}  "
              f"smo-sklearn={acc_smo - acc_ref:+.4f}  "
              f"scipy-sklearn={acc_scipy - acc_ref:+.4f}  "
              f"smo-scipy={acc_smo - acc_scipy:+.4f}")


def main():
    args = parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)

    report_path = os.path.join(REPORT_DIR, "check_gamma.txt")

    orig_stdout = sys.stdout
    with open(report_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(orig_stdout, log_file)
        try:
            print(f"[check_gamma] отчёт: {report_path}")
            print(f"[check_gamma] аргументы: {vars(args)}")
            run(args)
            print(f"\n[check_gamma] готово. "
                  f"отчёт сохранён: {report_path}")
        finally:
            sys.stdout = orig_stdout


if __name__ == "__main__":
    main()