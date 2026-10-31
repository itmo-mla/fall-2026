import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .pca import CustomPCA


def plot_explained_variance(custom_pca, sklearn_pca, eff_dim, filename="explained_variance.png"):
    """График объяснённой дисперсии: наша PCA vs sklearn PCA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    total_var = np.sum(custom_pca.explained_variance_)
    cum_custom = np.cumsum(custom_pca.explained_variance_) / total_var * 100
    cum_sklearn = np.cumsum(sklearn_pca.explained_variance_) / np.sum(sklearn_pca.explained_variance_) * 100

    n = len(custom_pca.explained_variance_)
    x = np.arange(1, n + 1)

    # Левый график: объяснённая дисперсия по компонентам
    axes[0].bar(x - 0.2, custom_pca.explained_variance_ / total_var * 100,
                width=0.35, label="Custom PCA", alpha=0.8, color='steelblue')
    axes[0].bar(x + 0.2, sklearn_pca.explained_variance_ / np.sum(sklearn_pca.explained_variance_) * 100,
                width=0.35, label="sklearn PCA", alpha=0.8, color='coral')
    axes[0].set_xlabel("Номер главной компоненты")
    axes[0].set_ylabel("Доля объяснённой дисперсии, %")
    axes[0].set_title("Дисперсия по компонентам")
    axes[0].set_xticks(x)
    axes[0].legend()

    # Правый график: кумулятивная дисперсия + критерий крутого склона
    axes[1].plot(x, cum_custom, 'o-', label="Custom PCA", color='steelblue', linewidth=2)
    axes[1].plot(x, cum_sklearn, 's--', label="sklearn PCA", color='coral', linewidth=2)
    axes[1].axhline(y=95, color='green', linestyle=':', alpha=0.7, label="Порог 95%")
    axes[1].axvline(x=eff_dim, color='red', linestyle='--', alpha=0.7,
                    label=f"Эфф. размерность m={eff_dim} (eps=0.05)")
    axes[1].set_xlabel("Количество компонент")
    axes[1].set_ylabel("Кумулятивная объяснённая дисперсия, %")
    axes[1].set_title("Кумулятивная дисперсия и эффективная размерность")
    axes[1].set_xticks(x)
    axes[1].legend(loc='lower right')
    axes[1].set_ylim([0, 105])

    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig(os.path.join('reports', filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_reconstruction_error(custom_pca, eff_dim, filename="reconstruction_error.png"):
    """Ошибка восстановления в зависимости от m."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n = len(custom_pca.explained_variance_)
    total_var = np.sum(custom_pca.explained_variance_)
    errors = []
    for m in range(1, n + 1):
        err = np.sum(custom_pca.explained_variance_[m:]) / total_var
        errors.append(err * 100)

    x = np.arange(1, n + 1)
    ax.plot(x, errors, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel("Количество компонент m")
    ax.set_ylabel("Остаточная дисперсия E_m, %")
    ax.set_title("Ошибка восстановления ||GU^T - F||^2 / ||F||^2")
    ax.set_xticks(x)

    ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label="Порог eps = 5%")
    ax.axvline(x=eff_dim, color='green', linestyle=':', alpha=0.7,
               label=f"Эфф. размерность m = {eff_dim}")

    ax.legend()
    ax.set_ylim([0, max(errors) * 1.1])

    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig(os.path.join('reports', filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    custom_pca = CustomPCA(n_components=None)
    custom_pca.fit(X_train_scaled)

    print(f"Сингулярные числа: {np.round(custom_pca.singular_values_, 2)}")
    print(f"Объяснённые дисперсии: {np.round(custom_pca.explained_variance_, 4)}")

    total_var = np.sum(custom_pca.explained_variance_)
    print(f"Суммарная дисперсия: {total_var:.6f}")
    print(f"Доли по компонентам (%): {np.round(custom_pca.explained_variance_ / total_var * 100, 2)}")

    print("Определение эффективной размерности...")
    for eps in [0.01, 0.05, 0.10, 0.20]:
        m = custom_pca.effective_dimension(epsilon=eps)
        cum_pct = np.sum(custom_pca.explained_variance_[:m]) / total_var * 100
        print(f"eps = {eps:.2f}: m = {m}, кумулятивная дисперсия = {cum_pct:.2f}%")

    eff_dim = custom_pca.effective_dimension(epsilon=0.05)
    print(f"Эффективная размерность (eps=0.05): m = {eff_dim}")

    print("Сравнение с эталонной реализацией")
    sklearn_pca = PCA(n_components=None)
    sklearn_pca.fit(X_train_scaled)

    # Сравнение собственных значений
    var_diff = np.max(np.abs(custom_pca.explained_variance_ - sklearn_pca.explained_variance_))
    print(f"Макс. расхождение собственных значений: {var_diff:.2e}")

    print("Косинусное сходство компонент (|cos|):")
    for i in range(len(custom_pca.components_)):
        cos_sim = np.abs(np.dot(custom_pca.components_[i], sklearn_pca.components_[i]))
        print(f"PC{i+1}: {cos_sim:.10f}")

    G_custom = custom_pca.transform(X_test_scaled)
    G_sklearn = sklearn_pca.transform(X_test_scaled)

    for i in range(G_custom.shape[1]):
        corr = np.abs(np.corrcoef(G_custom[:, i], G_sklearn[:, i])[0, 1])
        print(f"PC{i+1}: {corr:.10f}")

    X_recon_custom = custom_pca.inverse_transform(custom_pca.transform(X_test_scaled))
    recon_err_custom = np.mean((X_test_scaled - X_recon_custom) ** 2)

    X_recon_sklearn = sklearn_pca.inverse_transform(sklearn_pca.transform(X_test_scaled))
    recon_err_sklearn = np.mean((X_test_scaled - X_recon_sklearn) ** 2)

    print("MSE восстановления (полное пространство):")
    print(f"Custom PCA: {recon_err_custom:.2e}")
    print(f"sklearn PCA: {recon_err_sklearn:.2e}")

    pca_partial_c = CustomPCA(n_components=eff_dim)
    pca_partial_c.fit(X_train_scaled)
    X_recon_partial_c = pca_partial_c.inverse_transform(pca_partial_c.transform(X_test_scaled))
    recon_err_partial_c = np.mean((X_test_scaled - X_recon_partial_c) ** 2)

    pca_partial_s = PCA(n_components=eff_dim)
    pca_partial_s.fit(X_train_scaled)
    X_recon_partial_s = pca_partial_s.inverse_transform(pca_partial_s.transform(X_test_scaled))
    recon_err_partial_s = np.mean((X_test_scaled - X_recon_partial_s) ** 2)

    print(f"MSE восстановления (m={eff_dim}):")
    print(f"Custom PCA: {recon_err_partial_c:.6f}")
    print(f"sklearn PCA: {recon_err_partial_s:.6f}")
    print(f"Разница: {abs(recon_err_partial_c - recon_err_partial_s):.2e}")

    plot_explained_variance(custom_pca, sklearn_pca, eff_dim)

    plot_reconstruction_error(custom_pca, eff_dim)
