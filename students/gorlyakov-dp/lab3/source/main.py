import numpy as np
from algorithms import solve_dual
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(30)

X, y = make_moons(n_samples=300, noise=0.2, random_state=30)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30, stratify=y
)

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

C = 1.0
lambdas, dual_value = solve_dual(X_train, y_train, C)

print(f"C = {C}, значение целевой функции: {dual_value:.6f}")
print(f"Диапазон lambda: [{lambdas.min():.6f}, {lambdas.max():.6f}]")
print(f"Остаток ограничения sum(lambda * y) = 0: {abs(np.dot(lambdas, y_train)):.2e}")
