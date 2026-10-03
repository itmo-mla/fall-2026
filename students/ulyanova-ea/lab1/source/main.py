from pathlib import Path
from time import perf_counter

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import LinearClassifierMomentum, LinearClassifierSpeediest, LinearClassifier_mixed_version


def load_data():
    path = kagglehub.dataset_download("mastershomya/banknote-authetication")
    df = pd.read_csv(f"{path}/data_banknote_authentication.txt").drop_duplicates().reset_index(drop=True)
    X = df.drop(columns="class").to_numpy()
    y = np.where(df["class"].to_numpy() == 1, 1, -1)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    scaler = StandardScaler()
    return (scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test),
            y_train, y_val, y_test)


def metrics(y, predicted):
    return {
        "Accuracy": accuracy_score(y, predicted),
        "Precision": precision_score(y, predicted, pos_label=1, zero_division=0),
        "Recall": recall_score(y, predicted, pos_label=1, zero_division=0),
        "F1": f1_score(y, predicted, pos_label=1, zero_division=0),
    }


def plot_q(models):
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, model in models.items():
        ax.plot(np.arange(1, len(model.Q_history) + 1), model.Q_history, label=name)
    ax.set(title="Рекуррентная оценка функционала качества Q",
           xlabel="Номер градиентного шага t", ylabel="Оценка функционала Qₜ")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_margins(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, model in models.items():
        margins = np.sort(model.margins(X_test, y_test))
        ax.plot(np.arange(1, len(margins) + 1), margins, linewidth=2, label=name)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Mᵢ(w) = 0: граница ошибок")
    ax.axhline(1, color="gray", linestyle=":", linewidth=1, label="Mᵢ(w) = 1: Q(Mᵢ) = 0")
    ax.set(title="Отступы объектов тестовой выборки",
           xlabel="Номер объекта после сортировки по отступу", ylabel="Отступ Mᵢ(w) = g(xᵢ, w)yᵢ")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    constructors = {
        "Momentum": LinearClassifierMomentum,
        "Speediest": LinearClassifierSpeediest,
        "Mixed": LinearClassifier_mixed_version,
    }
    models, rows = {}, []
    for method, cls in constructors.items():
        for init in ("correlation", "multistart"):
            name = f"{method} / {init}"
            model = cls(init=init)
            started = perf_counter()
            model.fit(X_train, y_train, X_val, y_val)
            elapsed = perf_counter() - started
            models[name] = model
            row = {"Model": name, "Fit time, s": elapsed}
            for split, xs, ys in (("Train", X_train, y_train),
                                  ("Validation", X_val, y_val),
                                  ("Test", X_test, y_test)):
                row.update({f"{split} {metric}": value for metric, value in metrics(ys, model.predict(xs)).items()})
            rows.append(row)

    reference = RidgeClassifier(alpha=1.0)
    started = perf_counter()
    reference.fit(X_train, y_train)
    row = {"Model": "RidgeClassifier (reference)", "Fit time, s": perf_counter() - started}
    for split, xs, ys in (("Train", X_train, y_train),
                          ("Validation", X_val, y_val),
                          ("Test", X_test, y_test)):
        row.update({f"{split} {metric}": value for metric, value in metrics(ys, reference.predict(xs)).items()})
    rows.append(row)

    print(pd.DataFrame(rows).set_index("Model").round(4).to_string())
    output_dir = Path(__file__).resolve().parent.parent / "img"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_q(models).savefig(output_dir / "q_history.png", dpi=160)
    plot_margins(models, X_test, y_test).savefig(output_dir / "margins.png", dpi=160)
    plt.close("all")


if __name__ == "__main__":
    main()
