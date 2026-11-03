import numpy as np
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
