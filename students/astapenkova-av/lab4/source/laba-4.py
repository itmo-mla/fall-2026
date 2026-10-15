import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

class PCA_SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        # Центрирование
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Сингулярное разложение
        U, S, Vt = np.linalg.svd(
            X_centered, full_matrices=False
        )

        # Собственные значения ковариационной матрицы
        self.explained_variance_ = S**2 / (X.shape[0] - 1)

        # Доля объясненной дисперсии
        self.explained_variance_ratio_ = (
            self.explained_variance_
            / self.explained_variance_.sum()
        )

        k = (
            len(S)
            if self.n_components is None
            else self.n_components
        )

        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    images_dir = base_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 1. Загрузка данных
    data = load_diabetes()
    X = data.data
    y = data.target

    print("Размерность датасета:", X.shape)

    # 2. Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Стандартизация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Собственная реализация PCA
    my_pca = PCA_SVD()
    my_pca.fit(X_train)

    # 5. Эффективная размерность
    cumulative_variance = np.cumsum(
        my_pca.explained_variance_ratio_
    )

    threshold = 0.95
    k_eff = np.searchsorted(
        cumulative_variance, threshold
    ) + 1

    print("Эффективная размерность:", k_eff)
    print(
        "Сохраненная дисперсия:",
        cumulative_variance[k_eff - 1]
    )

    # 6. График объясненной дисперсии
    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o",
        label="Накопленная объясненная дисперсия"
    )

    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Порог 95%"
    )

    plt.axvline(
        k_eff,
        color="green",
        linestyle="--",
        label=f"Эффективная размерность: {k_eff}"
    )

    plt.xlabel("Количество главных компонент")
    plt.ylabel("Доля объясненной дисперсии")
    plt.title("Определение эффективной размерности")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(images_dir / "explained_variance.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # 7. Сравнение с sklearn
    my_pca = PCA_SVD(n_components=k_eff)
    Z_my = my_pca.fit_transform(X_train)

    sk_pca = PCA(n_components=k_eff)
    Z_sk = sk_pca.fit_transform(X_train)

    # Сравниваем дисперсию
    print(
        "Совпадение объясненной дисперсии:",
        np.allclose(
            my_pca.explained_variance_[:k_eff],
            sk_pca.explained_variance_,
            rtol=1e-7,
            atol=1e-10
        )
    )

    # Знак главных компонент может отличаться.
    # Выравниваем знаки перед сравнением проекций.
    signs = np.sign(
        np.sum(my_pca.components_ * sk_pca.components_, axis=1)
    )
    signs[signs == 0] = 1

    Z_my_aligned = Z_my * signs

    print(
        "Совпадение проекций:",
        np.allclose(
            Z_my_aligned,
            Z_sk,
            rtol=1e-5,
            atol=1e-8
        )
    )

    # 8. Линейная регрессия без PCA
    reg_original = LinearRegression()
    reg_original.fit(X_train, y_train)

    pred_original = reg_original.predict(X_test)

    # Линейная регрессия после PCA
    Z_test = my_pca.transform(X_test)

    reg_pca = LinearRegression()
    reg_pca.fit(Z_my, y_train)

    pred_pca = reg_pca.predict(Z_test)

    print("\nРегрессия без PCA:")
    print("MSE:", mean_squared_error(y_test, pred_original))
    print("R2:", r2_score(y_test, pred_original))

    print("\nРегрессия после PCA:")
    print("MSE:", mean_squared_error(y_test, pred_pca))
    print("R2:", r2_score(y_test, pred_pca))

    # 9. Сравнение предсказаний
    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_test,
        pred_original,
        alpha=0.6,
        label="Без PCA"
    )

    plt.scatter(
        y_test,
        pred_pca,
        alpha=0.6,
        label="После PCA"
    )

    limits = [
        min(y_test.min(), pred_original.min(), pred_pca.min()),
        max(y_test.max(), pred_original.max(), pred_pca.max())
    ]

    plt.plot(
        limits,
        limits,
        "k--",
        label="Идеальное предсказание"
    )

    plt.xlabel("Истинное значение")
    plt.ylabel("Предсказанное значение")
    plt.title("Сравнение линейной регрессии")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(images_dir / "regression_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()