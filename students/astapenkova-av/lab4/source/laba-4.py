# Еслм не открывается датасет, то можно использовать в теминале "open "/Applications/Python 3.14/Install Certificates.command"
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def effective_dimension(lam, eps=0.05):
    total = np.sum(lam)

    for m in range(1, len(lam) + 1):
        Em = np.sum(lam[m:]) / total

        if Em <= eps:
            return m

def PCA(X, eps=0.05):
    mean = np.mean(X, axis=0)
    F = X - mean

    _ , sigma, Ut = np.linalg.svd(F, full_matrices=False)
    U = Ut.T
    lam = sigma ** 2
    explain_variance_ratio = lam / np.sum(lam)
    comulative_evr = np.cumsum(explain_variance_ratio)
    m = effective_dimension(lam, eps)
    return mean, U, sigma, explain_variance_ratio, comulative_evr, m

# это для себя, можно не читать: w = U D^+ V^T (y - mean(y))
def ols_svd(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    F, yc = X - x_mean, y - y_mean
    V, sigma, Ut = np.linalg.svd(F, full_matrices=False)
    rhs = V.T @ yc
    tol = np.finfo(float).eps * max(F.shape) * sigma[0]
    beta = np.divide(rhs, sigma, out=np.zeros_like(rhs), where=sigma > tol)
    w = Ut.T @ beta
    return w, float(y_mean - x_mean @ w)

# w_tau = U diag(sigma/(sigma^2+tau)) V^T yс
def ridge_svd_from_decomposition(V, sigma, Ut, yc, tau):
    beta = sigma / (sigma**2 + tau) * (V.T @ yc)
    return Ut.T @ beta

def ridge_svd(X, y, tau):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    F, yc = X - x_mean, y - y_mean
    V, sigma, Ut = np.linalg.svd(F, full_matrices=False)
    w = ridge_svd_from_decomposition(V, sigma, Ut, yc, tau)
    return w, float(y_mean - x_mean @ w)

def save_current(out_dir, name):
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close()

def rmse(y, prediction):
    return np.sqrt(mean_squared_error(y, prediction))

def main():
    california = fetch_california_housing(as_frame=True)
    X, y = california.data, california.target
    names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    out = Path(__file__).resolve().parent.parent / "images"
    out.mkdir(parents=True, exist_ok=True)

    def save(name):
        plt.tight_layout()
        plt.savefig(out / f"{name}.png", dpi=150)
        plt.close()

    # PCA через SVD и эталонная реализация
    mean, U, sigma, explain_variance_ratio, comulative_evr, m = PCA(X_train)
    F = X_train - mean
    G = F @ U[:, :m]
    F_hat = G @ U[:, :m].T

    pca_ref = SklearnPCA(n_components=0.95, svd_solver="full")
    G_ref = pca_ref.fit_transform(X_train)
    F_ref = pca_ref.inverse_transform(G_ref)
    print("Эффективная размерность:", m)
    print("Размерность sklearn:", pca_ref.n_components_)
    print("Восстановление совпадает:", np.allclose(F_hat + mean, F_ref))

    indices = np.arange(1, len(sigma) + 1)

    plt.plot(indices, sigma, marker="o")
    plt.title("Сингулярные значения")
    plt.xlabel("Номер компоненты")
    save("01_singular_values")

    plt.bar(indices, explain_variance_ratio * 100)
    plt.title("Дисперсия каждой компоненты")
    plt.xlabel("Номер компоненты")
    plt.ylabel("Доля дисперсии, %")
    save("02_explained_variance")

    plt.plot(indices, comulative_evr * 100, marker="o")
    plt.axhline(95, linestyle="--", label="95%")
    plt.axvline(m, linestyle=":", label=f"m={m}")
    plt.title("Накопленная объяснённая дисперсия")
    plt.xlabel("Количество компонент")
    plt.ylabel("Дисперсия, %")
    plt.legend()
    save("03_cumulative_variance")

    errors = 1 - np.r_[0, comulative_evr]
    plt.plot(np.arange(len(errors)), errors * 100, marker="o")
    plt.axhline(5, linestyle="--", label="Порог 5%")
    plt.title("Ошибка восстановления PCA")
    plt.xlabel("Количество компонент")
    plt.ylabel("Потерянная дисперсия, %")
    plt.legend()
    save("04_reconstruction_error")

    scores = F @ U
    plt.scatter(scores[:, 0], scores[:, 1], c=y_train, s=5, alpha=0.3)
    plt.colorbar(label="Стоимость жилья")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Данные в пространстве первых двух компонент")
    save("05_pc1_pc2")

    plt.imshow(U, aspect="auto", vmin=-1, vmax=1)
    plt.xticks(range(len(names)), [f"PC{i}" for i in indices])
    plt.yticks(range(len(names)), names)
    plt.colorbar(label="Вес исходного признака")
    plt.title("Коэффициенты главных компонент")
    save("06_pca_weights")

    # PCA + линейная регрессия: динамика качества при разных m
    pcr_train, pcr_test = [], []
    for k in indices:
        Z_train = F @ U[:, :k]
        Z_test = (X_test - mean) @ U[:, :k]
        reg = LinearRegression().fit(Z_train, y_train)
        pcr_train.append(rmse(y_train, reg.predict(Z_train)))
        pcr_test.append(rmse(y_test, reg.predict(Z_test)))

    plt.plot(indices, pcr_train, marker="o", label="Обучение")
    plt.plot(indices, pcr_test, marker="o", label="Тест")
    plt.axvline(m, linestyle="--", label=f"PCA: m={m}")
    plt.xlabel("Количество главных компонент")
    plt.ylabel("RMSE")
    plt.title("Качество регрессии после PCA")
    plt.legend()
    save("07_pcr_rmse")

    # Обычный МНК и Ridge через одно SVD
    y_mean = y_train.mean()
    V, s, Ut = np.linalg.svd(F, full_matrices=False)
    rhs = V.T @ (y_train - y_mean)

    def weights(tau):
        return Ut.T @ ((s / (s**2 + tau)) * rhs)

    w_ols = weights(0)
    taus = np.logspace(-2, 5, 30)
    ridge_weights = np.array([weights(tau) for tau in taus])
    tau = 1000.0  # Пример; не подобранный оптимум
    w_ridge = weights(tau)

    plt.figure(figsize=(9, 5))
    for j, name in enumerate(names):
        plt.plot(taus, ridge_weights[:, j], label=name)
    plt.xscale("log")
    plt.xlabel("Параметр Ridge (tau)")
    plt.ylabel("Коэффициент")
    plt.title("Как Ridge уменьшает коэффициенты")
    plt.legend(ncol=2, fontsize=8)
    save("08_ridge_weights")

    # Lasso: коэффициенты и отбор признаков
    alphas = np.logspace(-4, 0, 30)
    lasso_weights = np.array([
        Lasso(alpha=a, max_iter=20000).fit(X_train, y_train).coef_
        for a in alphas
    ])

    plt.figure(figsize=(9, 5))
    for j, name in enumerate(names):
        plt.plot(alphas, lasso_weights[:, j], label=name)
    plt.xscale("log")
    plt.xlabel("Параметр Lasso (alpha)")
    plt.ylabel("Коэффициент")
    plt.title("Как Lasso обнуляет коэффициенты")
    plt.legend(ncol=2, fontsize=8)
    save("09_lasso_weights")

    plt.plot(alphas, np.count_nonzero(lasso_weights, axis=1), marker="o")
    plt.xscale("log")
    plt.xlabel("Параметр Lasso (alpha)")
    plt.ylabel("Число ненулевых коэффициентов")
    plt.title("Отбор признаков Lasso")
    save("10_lasso_feature_selection")

    # Финальное сравнение на тестовых данных (параметры фиксированы заранее)
    lasso = Lasso(alpha=0.01, max_iter=20000).fit(X_train, y_train)
    pcr = LinearRegression().fit(G, y_train)
    results = {
        "МНК SVD": X_test @ w_ols + y_mean - mean @ w_ols,
        "Ridge SVD": X_test @ w_ridge + y_mean - mean @ w_ridge,
        "Lasso": lasso.predict(X_test),
        "PCA + МНК": pcr.predict((X_test - mean) @ U[:, :m]),
    }
    scores = {name: rmse(y_test, pred) for name, pred in results.items()}
    for name, error in scores.items():
        print(f"{name}: RMSE = {error:.4f}")

    plt.bar(scores.keys(), scores.values())
    plt.ylabel("RMSE на тестовой выборке")
    plt.title("Сравнение моделей")
    plt.xticks(rotation=15)
    save("11_test_rmse")

if __name__ == "__main__":
    main()
