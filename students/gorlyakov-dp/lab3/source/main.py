import numpy as np
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
