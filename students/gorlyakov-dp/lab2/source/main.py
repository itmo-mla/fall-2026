from pathlib import Path

import numpy as np
import pandas as pd
from algorithms import loo, predict_knn, predict_parzen
from metrics import accuracy_score, f1_score, precision_score, recall_score
from plots import plot_data_projection, plot_loo_risk
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

figures = Path(__file__).resolve().parent.parent / "figures"
figures.mkdir(exist_ok=True)
plot_data_projection(X_train, y_train, figures / "data_projection.png")

ks = np.arange(1, len(X_train) - 1)
risks = loo(X_train, y_train, ks)
best_index = np.argmin(risks)
k = ks[best_index]

print(f"k по LOO: {k}, доля ошибок: {risks[best_index]:.4f}")

plot_loo_risk(ks, risks, k, figures / "loo_risk.png")

reference = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
reference.fit(X_train, y_train)

predictions = {
    f"kNN, k={k}": predict_knn(X_train, y_train, X_test, k),
    f"kNN + Parzen, k={k}": predict_parzen(X_train, y_train, X_test, k),
    f"sklearn KNN, k={k}": reference.predict(X_test),
}

results = []
for name, y_pred in predictions.items():
    results.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
    )

print(pd.DataFrame(results).to_string(index=False))
