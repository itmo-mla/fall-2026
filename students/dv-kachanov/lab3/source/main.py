import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from plots import plot_decision_boundary, plot_margins, plot_quality
from svm_dual import DualSVM


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


# выбрать датасет для бинарной классификации;

X, y = make_moons(n_samples=160, noise=0.25, random_state=42)
y = 2 * y - 1

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


# реализовать решение двойственной задачи по лямбда;

C = 1.0
gamma = 0.5

linear_svm = DualSVM(C=C, kernel="linear").fit(X_train, y_train)
rbf_svm = DualSVM(C=C, kernel="rbf", gamma=gamma).fit(X_train, y_train)


# провернуть трюк с ядром;

rbf_prediction = rbf_svm.predict(X_test)


# построить линейный классификатор;

linear_prediction = linear_svm.predict(X_test)


# визуализировать решение;

plot_decision_boundary(
    linear_svm,
    X_train,
    y_train,
    OUTPUT_DIR / "linear_svm_boundary.png",
    "Линейный SVM"
)
plot_decision_boundary(
    rbf_svm,
    X_train,
    y_train,
    OUTPUT_DIR / "rbf_svm_boundary.png",
    "SVM с гауссовским ядром"
)
plot_margins(
    rbf_svm.margins(X_test, y_test),
    OUTPUT_DIR / "rbf_margins.png"
)


# сравнить с эталонным решением;

reference_model = SVC(C=C, kernel="rbf", gamma=gamma)
reference_model.fit(X_train, y_train)
reference_prediction = reference_model.predict(X_test)

results = [
    evaluate(y_test, linear_prediction, "Dual SVM linear"),
    evaluate(y_test, rbf_prediction, "Dual SVM RBF"),
    evaluate(y_test, reference_prediction, "sklearn SVC RBF"),
]

results_df = pd.DataFrame(results)
plot_quality(
    results_df["model"].to_list(),
    results_df["accuracy"].to_numpy(),
    OUTPUT_DIR / "svm_quality.png"
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("C:", C)
print("gamma:", gamma)
print("Linear support vectors:", int(linear_svm.support_mask.sum()))
print("RBF support vectors:", int(rbf_svm.support_mask.sum()))
print("RBF margins min/mean/max:", rbf_svm.margins(X_test, y_test).min(), rbf_svm.margins(X_test, y_test).mean(), rbf_svm.margins(X_test, y_test).max())
print()
print("Итоговые результаты:")
print(results_df.to_string(index=False))
