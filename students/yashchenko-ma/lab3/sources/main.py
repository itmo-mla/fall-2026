"""Лабораторная №3. SVM: запуск всех шагов. Нужен ai4i2020.csv рядом (или передать путь аргументом)."""
import sys
import numpy as np
import pandas as pd

from data import get_data
from compare import fit_own, fit_sklearn, results_table, print_agreement
from visualize import visualize

np.random.seed(42)


def main():
    # 1. датасет + предобработка
    X_train, X_test, y_train, y_test = get_data()

    # 2-4. собственная реализация: линейный классификатор и трюк с ядром (RBF)
    svm_lin, pred_own_lin, t_own_lin = fit_own(X_train, y_train, X_test, y_test, 'own linear', C=1.0, kernel='linear')
    svm_rbf, pred_own_rbf, t_own_rbf = fit_own(X_train, y_train, X_test, y_test, 'own rbf', C=1.0, kernel='rbf', gamma=0.2)

    # 6. эталон sklearn
    sk_lin, pred_sk_lin, t_sk_lin = fit_sklearn(X_train, y_train, X_test, y_test, 'linear', C=1.0, kernel='linear')
    sk_rbf, pred_sk_rbf, t_sk_rbf = fit_sklearn(X_train, y_train, X_test, y_test, 'rbf', C=1.0, kernel='rbf', gamma=0.2)

    pd.set_option('display.width', 200)
    print(results_table(y_test, [
        ('own (linear)', pred_own_lin, svm_lin.n_sv_, t_own_lin),
        ('sklearn (linear)', pred_sk_lin, sk_lin.n_support_.sum(), t_sk_lin),
        ('own (rbf)', pred_own_rbf, svm_rbf.n_sv_, t_own_rbf),
        ('sklearn (rbf)', pred_sk_rbf, sk_rbf.n_support_.sum(), t_sk_rbf),
    ]))
    print_agreement(pred_own_lin, pred_sk_lin, pred_own_rbf, pred_sk_rbf)

    # 5. визуализация (PCA 2D)
    visualize(X_train, y_train, C=1.0, gamma_rbf=0.5)


if __name__ == "__main__":
    main()