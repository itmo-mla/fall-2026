from pathlib import Path

import numpy as np
from algorithms import fit_pca, select_pca_dimension
from plots import plot_dimension_selection
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data = load_wine()
target_index = data.feature_names.index("alcohol")

X = np.delete(data.data, target_index, axis=1)
y = data.data[:, target_index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30
)

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

pca_mean, components, singular_values = fit_pca(X_train)
epsilon = 0.05
m, errors = select_pca_dimension(singular_values, epsilon)

U_m = components[:, :m]

F_train = X_train - pca_mean
F_test = X_test - pca_mean
G_train = np.dot(F_train, U_m)
G_test = np.dot(F_test, U_m)

X_train_reconstructed = np.dot(G_train, U_m.T) + pca_mean
X_test_reconstructed = np.dot(G_test, U_m.T) + pca_mean

train_error = np.sum((X_train_reconstructed - X_train) ** 2) / np.sum(F_train**2)
test_error = np.sum((X_test_reconstructed - X_test) ** 2) / np.sum(F_test**2)

print(f"Сингулярные числа: {singular_values}")
print(f"Размерность при epsilon={epsilon}: {m}")
print(f"Размерность train: {X_train.shape} -> {G_train.shape}")
print(f"Размерность test: {X_test.shape} -> {G_test.shape}")
print(f"Относительная ошибка восстановления train: {train_error:.6f}")
print(f"Относительная ошибка восстановления test: {test_error:.6f}")

figures = Path(__file__).resolve().parent.parent / "figures"
figures.mkdir(exist_ok=True)
plot_dimension_selection(
    singular_values,
    errors,
    epsilon,
    m,
    figures / "dimension_selection.png",
)

pca = PCA(n_components=1 - epsilon, svd_solver="full")
pca.fit(X_train)
G_train_reference = pca.transform(X_train)
G_test_reference = pca.transform(X_test)
X_train_reference = pca.inverse_transform(G_train_reference)
X_test_reference = pca.inverse_transform(G_test_reference)

assert m == pca.n_components_

signs = np.where(np.sum(U_m * pca.components_.T, axis=0) < 0, -1, 1)
np.testing.assert_allclose(
    singular_values[:m], pca.singular_values_, rtol=0, atol=1e-10, equal_nan=False
)
np.testing.assert_allclose(
    G_train * signs, G_train_reference, rtol=0, atol=1e-10, equal_nan=False
)
np.testing.assert_allclose(
    G_test * signs, G_test_reference, rtol=0, atol=1e-10, equal_nan=False
)
np.testing.assert_allclose(
    X_train_reconstructed, X_train_reference, rtol=0, atol=1e-10, equal_nan=False
)
np.testing.assert_allclose(
    X_test_reconstructed, X_test_reference, rtol=0, atol=1e-10, equal_nan=False
)

reference_train_error = np.sum((X_train_reference - X_train) ** 2) / np.sum(F_train**2)
reference_test_error = np.sum((X_test_reference - X_test) ** 2) / np.sum(F_test**2)

print(f"Размерность sklearn PCA: {pca.n_components_}")
print(f"Относительная ошибка восстановления sklearn train: {reference_train_error:.6f}")
print(f"Относительная ошибка восстановления sklearn test: {reference_test_error:.6f}")
print("Проверки совпадения с sklearn PCA пройдены (atol=1e-10)")
