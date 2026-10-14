import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from pca import (
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

from plots import plot_projection, plot_reconstruction, plot_scree

# Выбор датасета для линейной регрессии
dataset = load_diabetes()
F_all = np.asarray(dataset.data, dtype=float)
y_all = np.asarray(dataset.target, dtype=float)

# Разделение на обучающую и контрольную выборки
F, Fc, y, yc = train_test_split(F_all, y_all, test_size=0.25, random_state=42)
ell, n = F.shape
k = Fc.shape[0]
print(f"l = {ell}, k = {k}, n = {n}")

# Спектр S = F^T F и SVD F = V D U^T
S = scatter_matrix(F)
lam = spectrum(F)
V, d, U, lam_svd = lecture_svd(F)
print("lambda (с.з. S = F^T F):", lam)
print("D (сингулярные числа, диагональ):", d)
print("||F||^2 = tr S =", float(np.trace(S)))

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

plot_scree(lam, np.array(errors), m, EPS, OUTPUT_DIR / "scree.png")
plot_projection(G, y, OUTPUT_DIR / "projection.png")
plot_reconstruction(np.array(errors), m, OUTPUT_DIR / "reconstruction.png")

# Гребневая регрессия: Q_tau(alpha) = ||F alpha - y||^2 + tau ||alpha||^2
TAU = 1.0
alpha_star = ridge_alpha_star(F, y, TAU)
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
Gc = Fc @ Um
beta_star = compressed_beta_star(G, y)
alpha_comp = Um @ beta_star
print("beta* =", beta_star)
print("Q на обучении (сжатая, m) =", control_functional(F, y, alpha_comp))
print("Q на контроле (сжатая, m) =", control_functional(Fc, yc, alpha_comp))
print("beta = U^T alpha, откл.:",
      float(np.max(np.abs(beta_star - Um.T @ alpha_comp))))