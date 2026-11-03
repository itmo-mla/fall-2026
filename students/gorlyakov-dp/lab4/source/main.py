import numpy as np
from algorithms import fit_pca
from sklearn.datasets import load_wine
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
m = 2
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
print(f"Размерность train: {X_train.shape} -> {G_train.shape}")
print(f"Размерность test: {X_test.shape} -> {G_test.shape}")
print(f"Относительная ошибка восстановления train: {train_error:.6f}")
print(f"Относительная ошибка восстановления test: {test_error:.6f}")
