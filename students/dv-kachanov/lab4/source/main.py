import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from pca import (
    align_signs,
    center,
    compressed_beta_star,
    control_functional,
    effective_dimension,
    lecture_svd,
    principal_components,
    reconstruct,
    relative_errors,
    ridge_alpha_star,
    ridge_functional,
    scatter_matrix,
    spectrum,
    squared_error,
)

from plots import (
    plot_projection,
    plot_reconstruction,
    plot_reference_comparison,
    plot_scree,
)

# Выбор датасета для линейной регрессии
dataset = load_diabetes()
F_all = np.asarray(dataset.data, dtype=float)
y_all = np.asarray(dataset.target, dtype=float)

# Разделение на обучающую и контрольную выборки
F_raw, Fc_raw, y_raw, yc_raw = train_test_split(
    F_all, y_all, test_size=0.25, random_state=42
)
ell, n = F_raw.shape
k = Fc_raw.shape[0]
print(f"l = {ell}, k = {k}, n = {n}")

# Центрирование: среднее считается по обучающей выборке и применяется к контролю
F, mu = center(F_raw)
Fc, _ = center(Fc_raw, mu)
ybar = float(np.mean(y_raw))
y = y_raw - ybar
yc = yc_raw - ybar

print("mu (средние признаков по обучению) =", mu)
print("Макс. |среднее по столбцам F| после центрирования:",
      float(np.max(np.abs(F.mean(axis=0)))))
print("ybar =", ybar)
print("Макс. |среднее y| после центрирования:", float(np.abs(y.mean())))

# Спектр S = F^T F и SVD F = V D U^T (F уже центрирована)
S = scatter_matrix(F)
lam = spectrum(F)
V, d, U, lam_svd = lecture_svd(F)
print("lambda (центрированные данные):", lam)
print("D (сингулярные числа, диагональ):", d)
print("||F||^2 = tr S =", float(np.trace(S)))

# Для сравнения: спектр без центрирования
lam_uncentered = spectrum(F_raw)
print("lambda (без центрирования, для сравнения):", lam_uncentered)

lam_diff = float(np.max(np.abs(lam - lam_svd)))
print("Макс. отклонение lambda (eig(S) vs SVD):", lam_diff)
print("Ортогональность U^T U = I, откл.:", float(np.max(np.abs(U.T @ U - np.eye(n)))))
print("Ортогональность V^T V = I, откл.:", float(np.max(np.abs(V.T @ V - np.eye(n)))))

# Тождества лекции: G^T G = Lambda диагональна
EPS = 0.05
errors = relative_errors(lam)
m = effective_dimension(lam, EPS)
print("E_m:", errors)
print(f"Выбранная размерность m = {m} (E_m <= eps = {EPS})")
print(f"Крутой склон: E_{m - 1} = {errors[m - 1]:.6f} >> E_{m} = {errors[m]:.6f} "
      f"(отношение {errors[m - 1] / errors[m]:.2f})")

Um = U[:, :m]
G = principal_components(F, U, m)
GtG = G.T @ G
print("Макс. внедиагональный элемент G^T G:", float(np.max(np.abs(GtG - np.diag(np.diag(GtG))))))
print("Макс. отклонение diag(G^T G) от lambda_1..lambda_m:",
      float(np.max(np.abs(np.diag(GtG) - lam[:m]))))

# Ошибка восстановления: прямое вычисление vs хвост lambda
Fhat = reconstruct(G, U, m)
abs_err = squared_error(F, Fhat)
tail = float(np.sum(lam[m:]))
print("||G U^T - F||^2 (прямо) =", abs_err)
print("lambda_{m+1} + ... + lambda_n (хвост) =", tail)

# Эталонный PCA из sklearn: он сам центрирует данные, поэтому сравниваем
# с нашей центрированной реализацией.
ref = PCA(n_components=n, svd_solver="full").fit(F_raw)
lam_ref = ref.explained_variance_ * (ell - 1)
d_ref = ref.singular_values_
U_ref = ref.components_.T
U_aligned, signs = align_signs(U, U_ref)
G_ref = ref.transform(F_raw) * signs
Gc = Fc @ Um
Gc_ref = ref.transform(Fc_raw)[:, :m] * signs[:m]
Fhat_ref = ref.inverse_transform(ref.transform(F_raw))
Fhat_raw = Fhat + mu
Fhat_full = reconstruct(principal_components(F, U, n), U, n)
ref_m = PCA(n_components=m, svd_solver="full").fit(F_raw)
Fhat_ref_m = ref_m.inverse_transform(ref_m.transform(F_raw))
errors_ref = relative_errors(lam_ref)

print("\n=== Сравнение с эталонным PCA (sklearn.decomposition.PCA) ===")
print("Макс. |mu - ref.mean_| =", float(np.max(np.abs(mu - ref.mean_))))
print("Макс. |lambda - lambda_ref| =", float(np.max(np.abs(lam - lam_ref))))
print("Макс. |d - ref.singular_values_| =", float(np.max(np.abs(d - d_ref))))
print("Макс. |E_m - E_m^ref| =", float(np.max(np.abs(np.array(errors) - np.array(errors_ref)))))
print("Макс. |U - U_ref| (после согласования знаков) =",
      float(np.max(np.abs(U_aligned - U_ref))))
print("Макс. |G - G_ref| =", float(np.max(np.abs(G - G_ref[:, :m]))))
print("Макс. |G' - G'_ref| (контроль) =",
      float(np.max(np.abs(Gc - Gc_ref))))
print("Макс. |Fhat_full + mu - ref.inverse_transform(...)| =",
      float(np.max(np.abs(Fhat_full + mu - Fhat_ref))))
print("Макс. |Fhat_m + mu - ref_m.inverse_transform(...)| =",
      float(np.max(np.abs(Fhat_raw - Fhat_ref_m))))
print("Макс. |explained_variance_ratio - ref| =",
      float(np.max(np.abs(lam / np.sum(lam) - ref.explained_variance_ratio_))))
print("Макс. относительное отклонение lambda:",
      float(np.max(np.abs(lam - lam_ref) / np.maximum(np.abs(lam_ref), 1e-15))))

plot_scree(lam, np.array(errors), m, EPS, OUTPUT_DIR / "scree.png")
plot_projection(G, y_raw, OUTPUT_DIR / "projection.png")
plot_reconstruction(np.array(errors), m, OUTPUT_DIR / "reconstruction.png")
plot_reference_comparison(
    lam, lam_ref, errors, errors_ref, m, OUTPUT_DIR / "comparison.png"
)

# Гребневая регрессия: Q_tau(alpha) = ||F alpha - y||^2 + tau ||alpha||^2
# F и y центрированы, поэтому свободный член равен ybar.
TAU = 1.0
alpha_star = ridge_alpha_star(F, y, TAU)
print("\n=== Регрессия на центрированных данных ===")
print("alpha*_tau =", alpha_star)
print("Q_tau(alpha*_tau) на обучении =", ridge_functional(F, y, alpha_star, TAU))
print("Q(alpha*_tau, X^k) = ||F' alpha*_tau - y'||^2 на контроле =",
      control_functional(Fc, yc, alpha_star))
print("||alpha*_tau||^2 =", float(alpha_star @ alpha_star))

# Проверка формулы через SVD: alpha*_tau = U (D^2 + tau I)^{-1} D V^T y
alpha_svd = U @ (np.diag(d / (d ** 2 + TAU)) @ (V.T @ y))
print("Отклонение формулы SVD от прямого решения:",
      float(np.max(np.abs(alpha_svd - alpha_star))))

# Регрессия на сжатых признаках (слайд 24): ||G beta - y||^2 -> min
beta_star = compressed_beta_star(G, y)
alpha_comp = Um @ beta_star
print("beta* =", beta_star)
print("Q на обучении (сжатая, m) =", control_functional(F, y, alpha_comp))
print("Q на контроле (сжатая, m) =", control_functional(Fc, yc, alpha_comp))
print("beta = U^T alpha, откл.:",
      float(np.max(np.abs(beta_star - Um.T @ alpha_comp))))
