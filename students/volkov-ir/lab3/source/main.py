import time
import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing import prepare_data, PLOT_FEATURES
from model import calculate_gram, calculate_kernel, calculate_v, calculate_lambda, calculate_w, calculate_w0, calculate_w0_kernel, calculate_predict, calculate_fit_predict, calculate_grid_score
from metric import calculate_accuracy, calculate_margins, calculate_object_types
import plots
import sklearn_baseline

RUN_SELECTION = True # подбор C и gamma на валидации
RUN_PLOTS = True # картинки на двух признаках
RUN_SKLEARN = True # сравнение с SVC

eps = 1e-6

# 1 – данные
X_train, X_test, y_train, y_test, feature_names = prepare_data()
K_train = calculate_gram(X_train)
C = 1


# 2 – решение двойственной задачи на train
lam, count = calculate_lambda(K_train, y_train, C)

print("итераций:", count)
print("сумма lam * y:", lam @ y_train)
print("min lam:", np.min(lam), "max lam:", np.max(lam))
print("опорных:", np.sum(lam > eps), "из", len(lam))

w = calculate_w(lam, X_train, y_train)
w0 = calculate_w0(lam, X_train, y_train, w, C, eps)
print("w0, у SVC 0.0504:", w0)

# типы объектов по lam и их отступы
M = y_train * (X_train @ w - w0)
mask_periphery, mask_border, mask_violators = calculate_object_types(lam, C, eps)

print("периферийные:", np.sum(mask_periphery), "M от", np.min(M[mask_periphery]), "до", np.max(M[mask_periphery]))
print("граничные:", np.sum(mask_border), "M от", np.min(M[mask_border]), "до", np.max(M[mask_border]))
print("нарушители:", np.sum(mask_violators), "M от", np.min(M[mask_violators]), "до", np.max(M[mask_violators]))
# среди нарушителей M < 0 - ошибки классификации на train
print("из них ошибок:", np.sum(M[mask_violators] < 0))


# 3 – с ядром

# проверка ядер на ручном наборе
# X_1 = np.array([[1, 2], [3, 0], [2, 3]])
# print("линейное, ожидается [[5 3 8] [3 9 6] [8 6 13]]:", calculate_kernel(X_1, X_1, "linear", 0, 0))
# print("poly d=2 для x1 и x2, ожидается 16:", calculate_kernel(X_1, X_1, "poly", 2, 1)[0, 1])
# print("rbf gamma=0.5, ожидается exp(-4) = 0.0183 вне диагонали в [0, 1] и 1 на диагонали:", calculate_kernel(X_1, X_1, "rbf", 0, 0.5))
# на train линейное ядро должно совпасть со старой матрицей Грама
# print("линейное совпадает с calculate_gram:", np.allclose(calculate_kernel(X_train, X_train, "linear", 0, 0), K_train))
# test x train - не квадратная
# print("размер K(test, train):", calculate_kernel(X_test, X_train, "rbf", 0, 0.5).shape)
# w0 через ядро: для линейного должно совпасть с w0 через w
# print("w0 через K, ожидается", w0, ":", calculate_w0_kernel(lam, K_train, y_train, C, eps))

# gamma как в sklearn по умолчанию: 1 / (число признаков * дисперсия)
gamma = 1 / (X_train.shape[1] * X_train.var())
print("gamma:", gamma)

# rbf на train
# K_rbf = calculate_kernel(X_train, X_train, "rbf", 0, gamma)
# lam_rbf, count_rbf = calculate_lambda(K_rbf, y_train, C)
# print("rbf итераций:", count_rbf, "опорных:", np.sum(lam_rbf > eps))
# print("rbf w0:", calculate_w0_kernel(lam_rbf, K_rbf, y_train, C, eps))

# подбор C и gamma: от train отрезаем валидацию
if RUN_SELECTION:
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)
    C_values = [0.1, 1, 10]
    gamma_values = [0.3 * gamma, gamma, 3 * gamma]

    for kernel in ["linear", "poly", "rbf"]:
        best = None
        for C_k in C_values:
            # у линейного ядра gamma нет
            for gamma_k in ([0] if kernel == "linear" else gamma_values):
                y_pred, lam_k = calculate_fit_predict(X_fit, y_fit, X_val, kernel, 2, C_k, gamma_k)
                acc = calculate_accuracy(y_pred, y_val)
                print(kernel, "C:", C_k, "gamma:", round(gamma_k, 4), "опорных:", np.sum(lam_k > eps), "точность на валидации:", round(acc, 3))
                if best is None or acc > best[0]:
                    best = (acc, C_k, gamma_k)
        # лучшие параметры: обучаемся на всем train, один раз проверяем на test
        y_pred, lam_k = calculate_fit_predict(X_train, y_train, X_test, kernel, 2, best[1], best[2])
        print("лучшее", kernel, "C:", best[1], "gamma:", round(best[2], 4), "опорных:", np.sum(lam_k > eps), "точность на test:", round(calculate_accuracy(y_pred, y_test), 3))


# 4 – линейный классификатор, лучшее C = 0.1, отступы на 30 признаках
C_best = 0.1
lam_best, count_best = calculate_lambda(K_train, y_train, C_best)
w0_best = calculate_w0_kernel(lam_best, K_train, y_train, C_best, eps)
M_best = calculate_margins(K_train, lam_best, y_train, w0_best)
mask_periphery, mask_border, mask_violators = calculate_object_types(lam_best, C_best, eps)
print("C = 0.1, периферийные:", np.sum(mask_periphery), "граничные:", np.sum(mask_border), "нарушители:", np.sum(mask_violators), "ошибок:", np.sum(M_best < 0))
y_pred = calculate_predict(calculate_kernel(X_test, X_train, "linear", 0, 0), lam_best, y_train, w0_best)
print("C = 0.1, точность на test:", round(calculate_accuracy(y_pred, y_test), 3))


# 5 – картинки на двух признаках
if RUN_PLOTS:
    plots.plot_margins(M_best, lam_best, C_best, "../images/margins.png")

    X2_train, X2_test, y2_train, y2_test, names_2d = prepare_data(features=PLOT_FEATURES)
    gamma_2d = 1 / (X2_train.shape[1] * X2_train.var())
    xx, yy, points = plots.make_grid(X2_train)

    # три ядра при C = 1
    panels = []
    for kernel, label in [("linear", "линейное, C = 1"), ("poly", f"poly d = 2, C = 1, gamma = {gamma_2d:.2f}"), ("rbf", f"RBF, C = 1, gamma = {gamma_2d:.2f}")]:
        Z, lam_2d, M_2d = calculate_grid_score(X2_train, y2_train, kernel, 2, 1, gamma_2d, points, xx.shape)
        panels.append((label, Z, lam_2d, 1, M_2d))
    plots.plot_boundaries(panels, xx, yy, X2_train, y2_train, names_2d, "Разделяющая граница для трех ядер", "../images/kernels.png")

    # влияние C, линейное ядро
    panels = []
    for C_k in [0.01, 1, 100]:
        Z, lam_2d, M_2d = calculate_grid_score(X2_train, y2_train, "linear", 2, C_k, gamma_2d, points, xx.shape)
        # ширина полосы 2 / ||w||, w считается только у линейного ядра
        width = 2 / np.linalg.norm(calculate_w(lam_2d, X2_train, y2_train))
        panels.append((f"C = {C_k}, ширина {width:.2f}", Z, lam_2d, C_k, M_2d))
    plots.plot_boundaries(panels, xx, yy, X2_train, y2_train, names_2d, "Влияние C, линейное ядро", "../images/influence_c.png")

    # влияние gamma, RBF при C = 1
    panels = []
    for gamma_k in [0.1 * gamma_2d, gamma_2d, 10 * gamma_2d]:
        Z, lam_2d, M_2d = calculate_grid_score(X2_train, y2_train, "rbf", 2, 1, gamma_k, points, xx.shape)
        panels.append((f"RBF, C = 1, gamma = {gamma_k:.2f}", Z, lam_2d, 1, M_2d))
    plots.plot_boundaries(panels, xx, yy, X2_train, y2_train, names_2d, "Влияние gamma", "../images/influence_gamma.png")


# 6 – сравнение с SVC, параметры из подбора на валидации
if RUN_SKLEARN:
    for kernel, C_k, gamma_k in [("linear", 0.1, 0), ("poly", 10, 0.01), ("rbf", 1, 0.01)]:
        K_tr = calculate_kernel(X_train, X_train, kernel, 2, gamma_k)
        start = time.time()
        lam_k, count_k = calculate_lambda(K_tr, y_train, C_k)
        end = time.time()
        w0_k = calculate_w0_kernel(lam_k, K_tr, y_train, C_k, eps)
        pred = calculate_predict(calculate_kernel(X_test, X_train, kernel, 2, gamma_k), lam_k, y_train, w0_k)

        lam_sk, w0_sk, pred_sk, time_sk = sklearn_baseline.calculate_sklearn_svm(X_train, y_train, X_test, kernel, 2, C_k, gamma_k)
        diff_lam, diff_w0, count_pred = sklearn_baseline.calculate_comparison(lam_k, lam_sk, w0_k, w0_sk, pred, pred_sk)

        print(kernel, "C:", C_k, "gamma:", gamma_k)
        print("  опорных:", np.sum(lam_k > eps), "у SVC:", np.sum(lam_sk > 0))
        print("  w0:", round(w0_k, 4), "у SVC:", round(w0_sk, 4), "расхождение:", round(diff_w0, 5))
        print("  наибольшее расхождение lam:", round(diff_lam, 5))
        print("  не совпало предсказаний на test:", count_pred, "из", len(y_test))
        print("  точность на test:", round(calculate_accuracy(pred, y_test), 3), "у SVC:", round(calculate_accuracy(pred_sk, y_test), 3))
        print("  время, с:", round(end - start, 2), "у SVC:", round(time_sk, 4))
