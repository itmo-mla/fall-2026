import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier, SGDClassifier

from dataset import load_data
from linear_classifier import LinearClassifier, margin, multistart
from metrics import classification_metrics, confusion_matrix
from plots import plot_comparison, plot_margins, plot_multistart, plot_training

IMAGES = Path(__file__).resolve().parent / "plots"

PARAMS = dict(lr=1e-3, momentum=0.9, tau=1e-3, n_epochs=100)
SEED = 0


def timed_fit(model, *args):
    start = time.perf_counter()
    model.fit(*args)
    return model, time.perf_counter() - start


def main():
    IMAGES.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, _ = load_data()
    print(f"train: {X_train.shape}, test: {X_test.shape}, "
          f"доля класса +1: {np.mean(y_train == 1):.2f}")
    fit_args = (X_train, y_train, X_test, y_test)

    models, times = {}, {}

    models["корреляция"], times["корреляция"] = timed_fit(
        LinearClassifier(init="correlation", seed=SEED, **PARAMS), *fit_args)

    start = time.perf_counter()
    best, risks = multistart(X_train, y_train, n_starts=10, seed=SEED,
                             X_val=X_test, y_val=y_test, **PARAMS)
    models["мультистарт (10 запусков)"] = best
    times["мультистарт (10 запусков)"] = time.perf_counter() - start
    print(f"мультистарт: риск min={min(risks):.4f}, max={max(risks):.4f}, лучший seed={best.seed}")

    models["случайный порядок"], times["случайный порядок"] = timed_fit(
        LinearClassifier(init="random", order="random", seed=SEED, **PARAMS), *fit_args)
    models["порядок по |M|"], times["порядок по |M|"] = timed_fit(
        LinearClassifier(init="random", order="margin", seed=SEED, **PARAMS), *fit_args)

    no_momentum = {**PARAMS, "momentum": 0.0}
    models["SGD без инерции"], times["SGD без инерции"] = timed_fit(
        LinearClassifier(init="random", seed=SEED, **no_momentum), *fit_args)
    models["скорейший спуск"], times["скорейший спуск"] = timed_fit(
        LinearClassifier(init="random", step="steepest", seed=SEED, **no_momentum), *fit_args)

    reference = {
        "sklearn RidgeClassifier": RidgeClassifier(alpha=1.0),
        "sklearn SGDClassifier": SGDClassifier(
            loss="squared_error", penalty="l2", alpha=PARAMS["tau"], learning_rate="constant",
            eta0=PARAMS["lr"], max_iter=PARAMS["n_epochs"], random_state=SEED),
    }
    for name, model in reference.items():
        _, times[name] = timed_fit(model, X_train[:, 1:], y_train)

    results = {}
    for name, model in models.items():
        results[name] = classification_metrics(y_test, model.predict(X_test))
    for name, model in reference.items():
        results[name] = classification_metrics(y_test, model.predict(X_test[:, 1:]))

    print("\n| Модель | Accuracy | Precision | Recall | F1 | Q (train) | Эпох | Время, с |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        if name in models:
            epochs = len(models[name].history["Q_epoch"])
            risk = f"{models[name].empirical_risk(X_train, y_train):.3f}"
        else:
            epochs, risk = "—", "—"
        print(f"| {name} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} "
              f"| {r['f1']:.3f} | {risk} | {epochs} | {times[name]:.2f} |")

    best_name = max(models, key=lambda n: (results[n]["f1"], results[n]["accuracy"]))
    print(f"\nлучшая собственная модель: {best_name}")
    print("матрица ошибок [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, models[best_name].predict(X_test)))

    corr = models["корреляция"]
    plot_margins({
        "Отступы после инициализации через корреляцию": margin(corr.w_init, X_train, y_train),
        "Отступы после обучения": margin(corr.w, X_train, y_train),
    }, IMAGES / "margins.png")
    plot_training({k: models[k] for k in ["корреляция", "мультистарт (10 запусков)", "случайный порядок"]},
                  IMAGES / "training_init.png", "Инициализация весов")
    plot_training({k: models[k] for k in ["случайный порядок", "порядок по |M|"]},
                  IMAGES / "training_order.png", "Порядок предъявления объектов")
    plot_training({"SGD с инерцией, γ = 0.9": models["случайный порядок"],
                   "SGD без инерции": models["SGD без инерции"],
                   "скорейший спуск, h* = 1/(2‖x‖²)": models["скорейший спуск"]},
                  IMAGES / "training_step.png", "Инерция и выбор шага")
    plot_multistart(risks, IMAGES / "multistart.png")
    plot_comparison(results, IMAGES / "comparison.png")
    print(f"\nграфики сохранены в {IMAGES}")


if __name__ == "__main__":
    main()
