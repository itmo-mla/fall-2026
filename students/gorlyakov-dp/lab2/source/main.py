import numpy as np
import pandas as pd
from algorithms import predict_parzen
from metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

np.random.seed(30)

data = load_breast_cancer()

X = data.data
y = np.where(data.target == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30, stratify=y
)

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

k = 3
y_pred = predict_parzen(X_train, y_train, X_test, k)

results = []
results.append(
    {
        "model": f"kNN + Parzen, k={k}",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
)

print(pd.DataFrame(results).to_string(index=False))
