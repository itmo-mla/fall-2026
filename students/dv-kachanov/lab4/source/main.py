import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pca import (
    MyPCA,
    full_spectrum,
    kaiser_dimension,
    reconstruction_error,
    variance_threshold_dimension,
)
from plots import plot_projection, plot_reconstruction, plot_scree


# выбрать датасет для линейной регрессии;

dataset = load_diabetes()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# реализовать PCA через сингулярное разложение;

eigenvalues, ratios = full_spectrum(X_train)


# определить эффективную размерность выборки;

kaiser_k = kaiser_dimension(eigenvalues)
threshold_k = variance_threshold_dimension(ratios, threshold=0.95)

effective_k = threshold_k

print("Собственные значения:", eigenvalues)
print("Доля объясненной дисперсии:", ratios)
print("Размерность по Кайзеру:", kaiser_k)
print("Размерность по порогу 0.95:", threshold_k)
print("Выбранная эффективная размерность:", effective_k)

plot_scree(eigenvalues, ratios, effective_k, OUTPUT_DIR / "scree.png")


# показать эквивалентность с эталонной реализацией;

own_model = MyPCA(n_components=effective_k).fit(X_train)

reference_model = PCA(n_components=effective_k, random_state=42)
reference_model.fit(X_train)

Z_own = own_model.transform(X_test)
Z_reference = reference_model.transform(X_test)

signs = np.sign(np.sum(own_model.components_ * reference_model.components_, axis=1))
signs[signs == 0] = 1.0
Z_aligned = Z_own * signs

print()
print("Макс. отклонение долей дисперсии:", np.max(np.abs(own_model.explained_variance_ratio_ - reference_model.explained_variance_ratio_)))
print("Макс. отклонение проекций:", np.max(np.abs(Z_aligned - Z_reference)))

plot_projection(own_model.transform(X_train), y_train, OUTPUT_DIR / "projection.png")


# ошибка восстановления при разных k;

n_features = X_train.shape[1]
errors = [reconstruction_error(X_train, k) for k in range(1, n_features + 1)]

plot_reconstruction(errors, effective_k, OUTPUT_DIR / "reconstruction.png")

print()
print("MSE восстановления при k*:", errors[effective_k - 1])
print("MSE восстановления при всех компонентах:", errors[-1])


# сравнить регрессию на полных и на сжатых признаках;

full_regression = Ridge(alpha=1.0).fit(X_train, y_train)
full_mse = mean_squared_error(y_test, full_regression.predict(X_test))

Z_train = own_model.transform(X_train)
compressed_regression = Ridge(alpha=1.0).fit(Z_train, y_train)
compressed_mse = mean_squared_error(y_test, compressed_regression.predict(Z_own))

results = pd.DataFrame([
    {"model": "Ridge на всех признаках", "mse": full_mse},
    {"model": f"Ridge на {effective_k} главных компонентах", "mse": compressed_mse},
])

print()
print("Итоговые результаты:")
print(results.to_string(index=False))


# подготовить отчет.
