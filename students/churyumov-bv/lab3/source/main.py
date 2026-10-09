# 1. выбрать датасет для бинарной классификации;
# 2. реализовать решение двойственной задачи по лямбда; для решения задачи использовать [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) или любую другую библиотеку;
# 3. провернуть трюк с ядром;
# 4. построить линейный классификатор;
# 5. визуализировать решение;
# 6. сравнить с эталонным решением;

import os
import time
import numpy as np
import pandas as pd
import kagglehub
import svm
import plots
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def path(filename):
    return os.path.join(OUTPUT_DIR, filename)

# 1. выбрать датасет для бинарной классификации;
def load_data():
    df = pd.read_csv(os.path.join(kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data"), "data.csv"))
    y = np.where(df["diagnosis"] == "M", 1, -1)                  # +1 - злокачественная опухоль, -1 - доброкачественная
    X = df.drop(columns=["id", "diagnosis", "Unnamed: 32"]).values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


# Три варианта ядра: (название, свой SVM-ядро по параметру, параметры эталона SVC, сетка параметров ядра)
KERNELS = {
    "linear": (lambda p: svm.linear_kernel(), lambda p: dict(kernel="linear"), [None]),
    "poly":   (lambda p: svm.poly_kernel(p), lambda p: dict(kernel="poly", degree=p, gamma=1, coef0=1), [2, 3]),
    "rbf":    (lambda p: svm.rbf_kernel(p), lambda p: dict(kernel="rbf", gamma=p), [0.003, 0.01, 0.03, 0.1]),
}
CS = [0.01, 0.1, 1, 10, 100]


def select_params(X, y):
    """Подбор C и параметра ядра по 5-fold CV. Перебор делается эталоном SVC: свой решатель (SLSQP) для сетки слишком медленный"""
    best, curves = {}, {}
    for name, (_, sk_params, grid) in KERNELS.items():
        for p in grid:
            gs = GridSearchCV(SVC(**sk_params(p)), {"C": CS}, cv=5, scoring="accuracy").fit(X, y)
            label = name if p is None else f"{name}, {'d' if name == 'poly' else 'γ'} = {p}"
            curves[label] = (CS, gs.cv_results_["mean_test_score"])
            if name not in best or gs.best_score_ > best[name][2]:
                best[name] = (p, gs.best_params_["C"], gs.best_score_)
    return best, curves


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # 2-4. своя реализация: двойственная задача, ядро, классификатор. Параметры - по CV
    best, curves = select_params(X_train, y_train)
    plots.plot_cv(curves, path("cv_grid.png"))
    print("Подбор параметров (5-fold CV):")
    for name, (p, C, score) in best.items():
        print(f"  {name:6s} param = {p}, C = {C}, CV accuracy = {score:.4f}")

    # 6. сравнение с эталоном на тех же параметрах
    models, refs = {}, {}
    print("\nСравнение с sklearn.svm.SVC на тестовой выборке:")
    print("kernel  | impl    | accuracy |   F1   | n_SV | time, s")
    for name, (p, C, _) in best.items():
        make_kernel, sk_params, _ = KERNELS[name]
        t = time.time()
        models[name] = svm.SVM(make_kernel(p), C=C).fit(X_train, y_train)
        t_own = time.time() - t
        t = time.time()
        refs[name] = SVC(C=C, **sk_params(p)).fit(X_train, y_train)
        t_ref = time.time() - t

        for impl, model, dt in (("own", models[name], t_own), ("sklearn", refs[name], t_ref)):
            y_pred = model.predict(X_test)
            n_sv = model.sv.sum() if impl == "own" else len(model.support_)
            print(f"{name:7s} | {impl:7s} | {accuracy_score(y_test, y_pred):.4f}   | {f1_score(y_test, y_pred):.4f} | {n_sv:4d} | {dt:.2f}")

        f_own, f_sk = models[name].decision_function(X_test), refs[name].decision_function(X_test)
        lam_sk = np.zeros(len(y_train))
        lam_sk[refs[name].support_] = np.abs(refs[name].dual_coef_[0])
        print(f"   совпадение предсказаний: {np.mean(models[name].predict(X_test) == refs[name].predict(X_test)):.4f}, "
              f"max|Δf| = {np.abs(f_own - f_sk).max():.4f}, max|Δλ| = {np.abs(models[name].lam - lam_sk).max():.4f}, "
              f"w0: {models[name].w0:.4f} vs {-refs[name].intercept_[0]:.4f}")
        if name == "rbf":
            plots.plot_compare(f_own, f_sk, models[name].lam, lam_sk, path("compare_rbf.png"))
            plots.plot_margins(models[name], path("margins_rbf.png"))

    # 5. визуализация: модели на проекции двух главных компонент (PCA обучается на train)
    pca = PCA(n_components=2).fit(X_train)
    Z_train = pca.transform(X_train)
    plots.plot_boundaries([
        ("линейное, C = 1", svm.SVM(svm.linear_kernel(), C=1).fit(Z_train, y_train)),
        ("полиномиальное, d = 3, C = 1", svm.SVM(svm.poly_kernel(3), C=1).fit(Z_train, y_train)),
        ("RBF, γ = 0.5, C = 1", svm.SVM(svm.rbf_kernel(0.5), C=1).fit(Z_train, y_train)),
    ], Z_train, y_train, path("kernels_pca.png"))

    plots.plot_boundaries([
        (f"C = {C}", svm.SVM(svm.linear_kernel(), C=C).fit(Z_train, y_train)) for C in (0.01, 1, 100)
    ], Z_train, y_train, path("c_influence_pca.png"))

    # линейно разделимая выборка (слайд 5): жёсткий зазор, сверяем w, w0 и lambda
    X_toy, y_toy = make_blobs(n_samples=60, centers=[[-2, -2], [2, 2]], cluster_std=0.8, random_state=RANDOM_STATE)
    y_toy = 2 * y_toy - 1
    toy = svm.SVM(svm.linear_kernel(), C=1e3).fit(X_toy, y_toy)
    toy_ref = SVC(kernel="linear", C=1e3).fit(X_toy, y_toy)
    lam_ref = np.zeros(len(y_toy))
    lam_ref[toy_ref.support_] = np.abs(toy_ref.dual_coef_[0])
    print("\nЛинейно разделимая выборка (жёсткий зазор, C = 1000):")
    print(f"  w  own = {toy.weights()}, sklearn = {toy_ref.coef_[0]}")
    print(f"  w0 own = {toy.w0:.5f}, sklearn = {-toy_ref.intercept_[0]:.5f}")
    print(f"  опорных: {toy.sv.sum()} vs {len(toy_ref.support_)}, max|Δλ| = {np.abs(toy.lam - lam_ref).max():.5f}")
    print(f"  ширина полосы 2/||w|| = {2 / np.linalg.norm(toy.weights()):.4f}, min M_i = {toy.margins().min():.4f}")
    plots.plot_boundaries([("линейно разделимая выборка, C = 1000", toy)], X_toy, y_toy, path("toy_hard_margin.png"),
                          xlabel="$x_1$", ylabel="$x_2$")
