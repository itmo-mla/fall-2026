import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from dataset import prepare_diabetes
from dual import gradient_mismatch
from figures import (
    plot_c_effect,
    plot_confusion_matrices,
    plot_data_projection,
    plot_single_boundary,
)
from kernel import default_gamma
from svm import SVM, SVMResult

C_LINEAR = 1.0
C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]


def _prepare_paths() -> tuple[Path, Path, Path]:
    source_dir = Path(__file__).resolve().parent
    lab_dir = source_dir.parent
    images_dir = lab_dir / "images"
    results_dir = source_dir / "res"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return lab_dir, images_dir, results_dir


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1 для положительного класса (+1)"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1,
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1,
                                     zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _n_support_of(model) -> int:
    """Число опорных векторов для собственной модели и для sklearn SVC"""
    if hasattr(model, "n_support"):
        return int(model.n_support)
    return int(model.support_vectors_.shape[0])


def _svm_result(key: str, label: str, model, y_train: np.ndarray,
                y_test: np.ndarray, X_train: np.ndarray,
                X_test: np.ndarray) -> SVMResult:
    """Собирает метрики и предсказания одной модели в SVMResult"""
    predictions = model.predict(X_test)
    return SVMResult(
        key=key,
        label=label,
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, pos_label=1,
                                        zero_division=0)),
        recall=float(recall_score(y_test, predictions, pos_label=1,
                                  zero_division=0)),
        f1=float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        n_support=_n_support_of(model),
        train_acc=float(accuracy_score(y_train, model.predict(X_train))),
        test_acc=float(accuracy_score(y_test, predictions)),
        predictions=predictions,
    )


def _pca_2d(X_train: np.ndarray, X_test: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Проекция на первые две главные компоненты (компоненты — по train)"""
    centered = X_train - X_train.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    explained = singular[:2] ** 2 / np.sum(singular ** 2)
    return (X_train @ components[:2].T), (X_test @ components[:2].T), explained


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    lab_dir, images_dir, results_dir = _prepare_paths()
    data = prepare_diabetes()
    X_train, y_train = data.X_train, data.y_train
    X_test, y_test = data.X_test, data.y_test

    print(f"Pima Indians Diabetes: {len(data.frame)} объектов, "
          f"{X_train.shape[1]} признаков, {len(data.class_names)} классов")
    print(f"train/test: {len(y_train)}/{len(y_test)}; классы train "
          f"{data.train_counts.tolist()}, test {data.test_counts.tolist()}")
    print(f"Записей с нулями в физических признаках: {data.zero_rows}")

    plot_data_projection(X_train, y_train, images_dir / "01_data_pca.png")

    # Линейное ядро: собственная модель против sklearn.
    svm_linear = SVM(C=C_LINEAR, kind="linear").fit(X_train, y_train)
    skl_linear = SVC(C=C_LINEAR, kernel="linear", tol=1e-6).fit(X_train, y_train)

    sum_lam = float(svm_linear.lambdas @ y_train)
    grad_diff = gradient_mismatch(svm_linear.lambdas, svm_linear.kernel_matrix)
    print(f"\nЛинейный SVM: ОВ={svm_linear.n_support}, Σλᵢyᵢ={sum_lam:.2e}, "
          f"расхождение градиента={grad_diff:.2e}")

    # RBF-ядро: собственная модель против sklearn (одинаковое gamma).
    gamma = default_gamma(X_train)
    print(f"RBF gamma = {gamma:.6f}")
    svm_rbf = SVM(C=C_LINEAR, kind="rbf", gamma=gamma).fit(X_train, y_train)
    skl_rbf = SVC(C=C_LINEAR, kernel="rbf", gamma=gamma, tol=1e-6
                  ).fit(X_train, y_train)
    sum_lam_rbf = float(svm_rbf.lambdas @ y_train)
    print(f"RBF SVM: ОВ={svm_rbf.n_support}, Σλᵢyᵢ={sum_lam_rbf:.2e}")

    self_linear = _svm_result("self_linear", "Свой Linear SVM", svm_linear,
                              y_train, y_test, X_train, X_test)
    skl_linear_res = _svm_result("sklearn_linear", "sklearn SVC (linear)",
                                 skl_linear, y_train, y_test, X_train, X_test)
    self_rbf = _svm_result("self_rbf", "Свой RBF SVM", svm_rbf,
                           y_train, y_test, X_train, X_test)
    skl_rbf_res = _svm_result("sklearn_rbf", "sklearn SVC (rbf)", skl_rbf,
                              y_train, y_test, X_train, X_test)

    linear_agreement = float(np.mean(self_linear.predictions ==
                                     skl_linear_res.predictions))
    rbf_agreement = float(np.mean(self_rbf.predictions ==
                                  skl_rbf_res.predictions))
    print(f"Совпадение Linear (свой/sklearn): {linear_agreement:.4f}")
    print(f"Совпадение RBF (свой/sklearn): {rbf_agreement:.4f}")

    # Перебор C для RBF с тёплым стартом по λ.
    c_results = []
    x0 = None
    for c in C_VALUES:
        model = SVM(C=c, kind="rbf", gamma=gamma).fit(X_train, y_train, x0=x0)
        x0 = model.lambdas
        c_results.append({
            "C": c,
            "train_acc": float(accuracy_score(y_train, model.predict(X_train))),
            "test_acc": float(accuracy_score(y_test, model.predict(X_test))),
            "support": model.n_support,
        })
        print(f"  C={c:6.2f}: train={c_results[-1]['train_acc']:.4f}, "
              f"test={c_results[-1]['test_acc']:.4f}, "
              f"ОВ={model.n_support}")

    # Границы на PCA-проекции (модели обучаются на 2D для наглядности).
    X_train_2d, X_test_2d, explained = _pca_2d(X_train, X_test)
    print(f"\nPCA: PC1={100 * explained[0]:.1f}%, PC2={100 * explained[1]:.1f}%")

    linear_2d = SVM(C=C_LINEAR, kind="linear").fit(X_train_2d, y_train)
    rbf_2d = SVM(C=C_LINEAR, kind="rbf", gamma=gamma).fit(X_train_2d, y_train)

    linear_2d_acc = float(accuracy_score(y_train, linear_2d.predict(X_train_2d)))
    rbf_2d_acc = float(accuracy_score(y_train, rbf_2d.predict(X_train_2d)))
    plot_single_boundary(
        linear_2d, X_train_2d, y_train,
        f"Линейный SVM на PCA-проекции\n"
        f"train acc = {linear_2d_acc:.3f}, ОВ = {linear_2d.n_support}",
        images_dir / "02_linear_boundary.png",
    )
    plot_single_boundary(
        rbf_2d, X_train_2d, y_train,
        f"RBF SVM на PCA-проекции\n"
        f"train acc = {rbf_2d_acc:.3f}, ОВ = {rbf_2d.n_support}",
        images_dir / "03_rbf_boundary.png",
    )

    plot_c_effect(c_results, images_dir / "05_c_effect.png")

    plot_confusion_matrices(
        y_test,
        {
            "Свой Linear": self_linear.predictions,
            "sklearn Linear": skl_linear_res.predictions,
            "Свой RBF": self_rbf.predictions,
            "sklearn RBF": skl_rbf_res.predictions,
        },
        images_dir / "06_confusion.png",
    )

    rows = [self_linear, skl_linear_res, self_rbf, skl_rbf_res]
    pd.DataFrame([
        {
            "model": r.label,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "n_support": r.n_support,
            "train_acc": r.train_acc,
            "test_acc": r.test_acc,
        }
        for r in rows
    ]).to_csv(results_dir / "model_metrics.csv", index=False,
              encoding="utf-8-sig")
    pd.DataFrame(c_results).to_csv(results_dir / "c_sweep.csv", index=False,
                                   encoding="utf-8-sig")

    print(f"\n{'=' * 52}")
    for r in rows:
        print(f"  {r.label}: acc={r.accuracy:.4f}, p={r.precision:.4f}, "
              f"r={r.recall:.4f}, f1={r.f1:.4f}, ОВ={r.n_support}")
    print(f"\nГрафики: {images_dir}")
    print(f"Таблицы: {results_dir}")


if __name__ == "__main__":
    main()
