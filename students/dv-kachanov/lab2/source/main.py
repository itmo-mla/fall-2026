import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from knn import VariableParzenKNN, select_k_by_loo
from plots import plot_loo_risks, plot_metric_comparison, plot_prototypes
from prototype_selection import ccv_risk, predict_by_nearest_prototype, select_prototypes


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# выбрать датасет для классификации;

dataset = load_wine()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# реализовать алгоритм KNN с методом окна Парзена переменной ширины;


# подобрать параметр k методом скользящего контроля (LOO);

k_values = np.arange(1, min(31, len(X_train) - 1))
best_k, loo_risks = select_k_by_loo(X_train, y_train, k_values)


# обосновать выбор параметров алгоритма, построить графики эмпирического риска для различных k;

plot_loo_risks(k_values, loo_risks, best_k, OUTPUT_DIR / "loo_risk.png")


# сравнить с эталонной реализацией KNN;

own_model = VariableParzenKNN(k=best_k).fit(X_train, y_train)
own_prediction = own_model.predict(X_test)

reference_model = KNeighborsClassifier(n_neighbors=best_k)
reference_model.fit(X_train, y_train)
reference_prediction = reference_model.predict(X_test)


# реализовать алгоритм отбора эталонов;

prototype_indices = select_prototypes(X_train, y_train)
X_prototypes = X_train[prototype_indices]
y_prototypes = y_train[prototype_indices]

prototype_prediction = predict_by_nearest_prototype(X_prototypes, y_prototypes, X_test)


# подготовить визуализацию результатов работы алгоритма отбора эталонов;

plot_prototypes(X_train, y_train, prototype_indices, OUTPUT_DIR / "prototypes.png")


# сравнить качество работы KNN с и без отбора эталонов;

results = [
    evaluate(y_test, own_prediction, "Variable Parzen KNN"),
    evaluate(y_test, prototype_prediction, "Prototype 1NN"),
    evaluate(y_test, reference_prediction, "sklearn KNeighborsClassifier"),
]

results_df = pd.DataFrame(results)

plot_metric_comparison(
    results_df["model"].to_list(),
    results_df["accuracy"].to_numpy(),
    OUTPUT_DIR / "quality_comparison.png"
)

print("Dataset:", dataset.DESCR.splitlines()[0])
print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Classes:", sorted(np.unique(y).tolist()))
print("Best k by LOO:", best_k)
print("LOO risk for best k:", loo_risks[np.where(k_values == best_k)[0][0]])
print("Prototype count:", len(prototype_indices))
print("Compression ratio:", len(prototype_indices) / len(X_train))
print("CCV risk for selected prototypes:", ccv_risk(X_train, y_train, prototype_indices))
print()
print("Итоговые результаты:")
print(results_df.to_string(index=False))


# подготовить небольшой отчет о проделанной работе.
