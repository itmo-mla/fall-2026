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
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()
    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}, "
          f"доля класса +1: {np.mean(y_train == 1):.2f}")
    fit_args = (X_train, y_train, X_val, y_val)

    models, times = {}, {}

    models["корреляция"], times["корреляция"] = timed_fit(
        LinearClassifier(init="correlation", seed=SEED, **PARAMS), *fit_args)

    start = time.perf_counter()
    best, risks = multistart(X_train, y_train, n_starts=10, seed=SEED,
                             X_val=X_val, y_val=y_val, **PARAMS)
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

    def predict(name, X):
        return models[name].predict(X) if name in models else reference[name].predict(X[:, 1:])

    val_results = {name: classification_metrics(y_val, predict(name, X_val)) for name in times}
    results = {name: classification_metrics(y_test, predict(name, X_test)) for name in times}

    print("\n| Модель | Accuracy (val) | F1 (val) | Accuracy | Precision | Recall | F1 | Q (train) | Эпох | Время, с |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        if name in models:
            epochs = len(models[name].history["Q_epoch"])
            risk = f"{models[name].empirical_risk(X_train, y_train):.3f}"
        else:
            epochs, risk = "-", "-"
        v = val_results[name]
        print(f"| {name} | {v['accuracy']:.3f} | {v['f1']:.3f} | {r['accuracy']:.3f} | {r['precision']:.3f} "
              f"| {r['recall']:.3f} | {r['f1']:.3f} | {risk} | {epochs} | {times[name]:.2f} |")

    best_name = max(models, key=lambda n: (val_results[n]["f1"], val_results[n]["accuracy"]))
    print(f"\nлучшая собственная модель по валидации: {best_name}")
    print(f"её качество на test: accuracy={results[best_name]['accuracy']:.3f}, "
          f"F1={results[best_name]['f1']:.3f}")
    print("матрица ошибок на test [[TN, FP], [FN, TP]]:")
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
                   "скорейший спуск": models["скорейший спуск"]},
                  IMAGES / "training_step.png", "Инерция и выбор шага", per_epoch=True)
    plot_multistart(risks, IMAGES / "multistart.png")
    plot_comparison(results, IMAGES / "comparison.png")
    print(f"\nграфики сохранены в {IMAGES}")


if __name__ == "__main__":
    main()
