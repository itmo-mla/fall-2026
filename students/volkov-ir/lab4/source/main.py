import numpy as np
from data_preprocessing import prepare_data
import model
import metric
import sklearn_baseline
import plots

# EPS_LIST - пороги потери E_m для эффективной размерности
# M_SKLEARN - число направлений, при котором сравниваем восстановление со sklearn
# SAVE_PLOTS - сохранять ли графики в images
EPS_LIST = [0.2, 0.1, 0.05, 0.01]
M_SKLEARN = 3
SAVE_PLOTS = False

# вывод в консоль
def print_title(title):
    print(title)


def print_data(X_train, X_test):
    print_title("1. Данные")
    print(f"train {X_train.shape}, test {X_test.shape}")
    print()


# loss_list[m - 1] – ошибка восстановления при m направлениях
def print_pca(lam, loss_list):
    print_title("2. PCA через SVD")
    shares = lam / np.sum(lam)
    print(f"{'направление':<13}{'lambda':>10}{'доля':>10}{'накопленная доля':>20}")
    for j in range(len(lam)):
        print(f"{j + 1:<13}{lam[j]:>10.1f}{shares[j]:>10.3f}{np.sum(shares[:j + 1]):>20.3f}")
    print()
    print(f"{'m':<4}{'ошибка восстановления':>24}")
    for m in range(1, len(lam) + 1):
        print(f"{m:<4}{loss_list[m - 1]:>24.1f}")
    print()


def print_effective_dim(e_m, eps_list, m_eps_list):
    print_title("3. Эффективная размерность")
    print(f"{'m':<14}" + "".join(f"{m:>8}" for m in range(len(e_m))))
    print(f"{'E_m':<14}" + "".join(f"{e:>8.3f}" for e in e_m))
    ratios = e_m[:-2] / e_m[1:-1]
    print(f"{'E_(m-1)/E_m':<14}" + f"{'':>8}" + "".join(f"{r:>8.2f}" for r in ratios))
    print()
    for i in range(len(eps_list)):
        print(f"порог eps = {eps_list[i]:<5} m = {m_eps_list[i]}")
    print()


# rows – список (название, наши значения, значения sklearn, совпало ли по всем элементам), показаны первые три значения
def print_sklearn(rows):
    print_title("4. Сравнение со sklearn")
    print(f"{'величина':<44}{'наши':<30}{'sklearn':<30}{'совпало'}")
    for name, own, sk, match in rows:
        print(f"{name:<44}{' '.join(f'{v:.3f}' for v in own[:3]):<30}{' '.join(f'{v:.3f}' for v in sk[:3]):<30}{'да' if match else 'нет'}")
    print()


def print_regression(rmse_list, r2_list, rmse_sk, r2_sk):
    print_title("5. Регрессия на главных компонентах")
    print(f"{'m':<24}{'RMSE на test':>14}{'R2 на test':>12}")
    for m in range(1, len(rmse_list) + 1):
        name = str(m)
        # при всех направлениях регрессия на главных компонентах совпадает с обычной
        if m == len(rmse_list) and np.isclose(rmse_list[-1], rmse_sk) and np.isclose(r2_list[-1], r2_sk):
            name = name + " (= LinearRegression)"
        print(f"{name:<24}{rmse_list[m - 1]:>14.3f}{r2_list[m - 1]:>12.3f}")
    print()


# 1 – данные
X_train, X_test, y_train, y_test, feature_names = prepare_data()
n_features = X_train.shape[1]
print_data(X_train, X_test)

# 2 – PCA через SVD
lam, u = model.calculate_pca(X_train)

# ошибка восстановления должна совпасть с суммой отброшенных lambda
loss_list = []
for m in range(1, n_features + 1):
    g = model.calculate_g(X_train, u, m)
    f_hat = model.calculate_f_hat(g, u, m)
    loss = model.calculate_loss(X_train, f_hat)
    loss_list.append(loss)
print_pca(lam, loss_list)

# 3 – эффективная размерность
e_m = model.calculate_e_m(lam)
m_eps_list = []
for eps in EPS_LIST:
    m_eps_list.append(model.calculate_m_eps(e_m, eps))
m_slope = model.calculate_m_slope(e_m)
print_effective_dim(e_m, EPS_LIST, m_eps_list)

if SAVE_PLOTS:
    plots.plot_spectrum(lam, "../images/spectrum.png")
    plots.plot_e_m(e_m, "../images/effective_dim.png")

# 4 – сравнение со sklearn
sklearn_rows = sklearn_baseline.calculate_comparison(X_train, X_test, lam, u, M_SKLEARN)
print_sklearn(sklearn_rows)

# 5 – регрессия на главных компонентах
y_mean = np.mean(y_train)
y_c = y_train - y_mean
rmse_list = []
r2_list = []
for m in range(1, n_features + 1):
    g_train = model.calculate_g(X_train, u, m)
    w = model.calculate_w(g_train, lam, y_c)
    g_test = model.calculate_g(X_test, u, m)
    y_pred = model.forecast(g_test, w, y_mean)
    rmse = metric.calculate_rmse(y_test, y_pred)
    r2 = metric.calculate_r2(y_test, y_pred)
    rmse_list.append(rmse)
    r2_list.append(r2)

# эталон на исходных признаках
y_pred_sk, coef_sk = sklearn_baseline.calculate_sklearn_regression(X_train, y_train, X_test)
rmse_sk = metric.calculate_rmse(y_test, y_pred_sk)
r2_sk = metric.calculate_r2(y_test, y_pred_sk)

print_regression(rmse_list, r2_list, rmse_sk, r2_sk)

if SAVE_PLOTS:
    plots.plot_rmse(rmse_list, rmse_sk, "../images/rmse.png")
