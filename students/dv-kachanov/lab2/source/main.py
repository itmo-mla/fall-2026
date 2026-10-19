"""Лабораторная работа 2: KNN с окном Парзена переменной ширины + отбор Omega.

Обозначения лекции: X --- объекты, Y --- ответы, X^l = (x_i, y_i)_{i=1}^l,
X^L = X^l ⊔ X^k. Здесь X^L отождествлено с обучающей выборкой (тестовая
выборка отложена и в CCV не участвует).
"""

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

from knn import G, LOO, ParzenKNN, select_k_by_LOO
from plots import plot_LOO, plot_metric_comparison, plot_prototypes
from prototype_selection import CCV, Pi, predict_by_Omega, select_Omega


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# выбрать датасет для классификации;

dataset = load_wine()
X = dataset.data
y = dataset.target

X_l, X_k, y_l, y_k = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_l = scaler.fit_transform(X_l)
X_k = scaler.transform(X_k)


# реализовать алгоритм KNN с методом окна Парзена переменной ширины
# a(x; X^l, k, K), K = G --- гауссовское ядро G(r) = exp(−2 r^2);


# подобрать параметр k методом скользящего контроля LOO(k, X^l) → min_k;

k_values = np.arange(1, min(31, len(X_l) - 1))
k_star, LOOs = select_k_by_LOO(X_l, y_l, k_values, K=G)


# обосновать выбор параметров алгоритма, построить графики эмпирического риска;

plot_LOO(k_values, LOOs, k_star, OUTPUT_DIR / "loo_risk.png")


# сравнить с эталонной реализацией KNN (sklearn --- только эталон);

own_model = ParzenKNN(k=k_star, K=G).fit(X_l, y_l)
own_prediction = own_model.predict(X_k)

reference_model = KNeighborsClassifier(n_neighbors=k_star)
reference_model.fit(X_l, y_l)
reference_prediction = reference_model.predict(X_k)


# реализовать алгоритм отбора эталонов: X^L := X^l, L = l + k, l = L − 1, k = 1;

X_L, y_L = X_l, y_l
L = len(X_L)
Omega, ccv_Omega = select_Omega(X_L, y_L, l=L - 1, k=1)
Omega = sorted(Omega)
X_Omega = X_L[Omega]
y_Omega = y_L[Omega]

prototype_prediction = predict_by_Omega(X_Omega, y_Omega, X_k)

# профиль компактности относительно Omega и CCV(Omega) при длинах контроля 1..3;
Pi_Omega = [Pi(m, X_L, y_L, set(Omega)) for m in range(1, 6)]
CCVs = {k: CCV(set(Omega), X_L, y_L, l=L - k, k=k) for k in (1, 2, 3)}


# подготовить визуализацию результатов работы алгоритма отбора эталонов;

plot_prototypes(X_L, y_L, np.array(Omega), OUTPUT_DIR / "prototypes.png")


# сравнить качество работы KNN с и без отбора эталонов;

results = [
    evaluate(y_k, own_prediction, "Parzen a(x; X^l, k, K)"),
    evaluate(y_k, prototype_prediction, "1NN a(x; X^l ∩ Omega)"),
    evaluate(y_k, reference_prediction, "sklearn KNeighborsClassifier"),
]

results_df = pd.DataFrame(results)

plot_metric_comparison(
    results_df["model"].to_list(),
    results_df["accuracy"].to_numpy(),
    OUTPUT_DIR / "quality_comparison.png"
)

print("Dataset:", dataset.DESCR.splitlines()[0])
print("l = |X^l|:", len(X_l))
print("Test |X^k|:", len(X_k))
print("Y:", sorted(np.unique(y).tolist()))
print("k* = argmin LOO(k, X^l):", k_star)
print("LOO(k*, X^l) =", LOOs[int(np.where(k_values == k_star)[0][0])])
print("Pi(m, Omega), m = 1..5:", [round(v, 6) for v in Pi_Omega])
for k, value in CCVs.items():
    print(f"CCV(Omega), l = {L - k}, k = {k}: {value}")
print("|Omega|:", len(Omega))
print("|Omega| / L:", len(Omega) / L)
print()
print("Итоговые результаты:")
print(results_df.to_string(index=False))


# подготовить небольшой отчет о проделанной работе.
