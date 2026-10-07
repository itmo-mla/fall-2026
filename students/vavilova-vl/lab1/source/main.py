import csv
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_preparation import (
    X_train,
    X_test,
    X_train_raw,
    X_test_raw,
    y_train,
    y_test
)

from model import LinearClassifier


PROJECT_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = PROJECT_DIR / "graphs"
MODELS_DIR = PROJECT_DIR / "models"
GRAPHS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

LEARNING_RATE = 0.001
MOMENTUM = 0.9
L2_LAMBDA = 0.001
N_STARTS = 10
N_ITERATIONS = 5000
MAIN_MODEL_NAME = "Основная: Multistart SGD + Momentum + L2"

best_model, multistart_losses, primary_loss_history = LinearClassifier.fit_primary(
    X_train,
    y_train,
    n_starts=N_STARTS,
    learning_rate=LEARNING_RATE,
    momentum=MOMENTUM,
    l2_lambda=L2_LAMBDA,
    n_iterations=N_ITERATIONS,
)


def fit_metric_model(name, features, targets, random_state=42, track_loss=False):
    if name == "Корреляция + скорейший спуск":
        model = LinearClassifier(
            features.shape[1],
            initial_weights=LinearClassifier.correlation_initialization(
                features,
                targets,
            ),
        )
        history = model.fit_steepest_descent(features, targets, n_iterations=100)
    elif name == "Случайный порядок SGD":
        model = LinearClassifier(features.shape[1])
        model.random_initialization(random_state=random_state)
        history = model.fit_sgd_random_presentation(
            features,
            targets,
            learning_rate=LEARNING_RATE,
            n_epochs=40,
            random_state=random_state,
        )
    elif name == "Порядок по |отступу| SGD":
        model = LinearClassifier(features.shape[1])
        model.random_initialization(random_state=random_state)
        history = model.fit_sgd_margin(
            features,
            targets,
            learning_rate=LEARNING_RATE,
            n_epochs=40,
        )
    elif name == "Рекуррентная оценка качества":
        model = LinearClassifier(features.shape[1])
        model.random_initialization(random_state=random_state)
        history = model.fit_sgd_recursive_quality(
            features,
            targets,
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
            random_state=random_state,
        )
    elif name == "Эталон: LogisticRegression":
        model = LogisticRegression(
            fit_intercept=False,
            random_state=random_state,
            max_iter=1 if track_loss else 2000,
            warm_start=track_loss,
        )
        if track_loss:
            history = []
            previous_loss = float("inf")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                for _ in range(200):
                    model.fit(features, targets)
                    positive_class_index = list(model.classes_).index(1)
                    scores = (
                        2 * model.predict_proba(features)[:, positive_class_index]
                        - 1
                    )
                    loss = float(np.mean((targets - scores) ** 2) / 2)
                    history.append(loss)
                    if abs(previous_loss - loss) < 1e-12:
                        break
                    previous_loss = loss
        else:
            model.fit(features, targets)
            history = []
    else:
        raise ValueError(f"Неизвестный вариант для сравнения: {name}")

    return model, history


METRIC_MODEL_NAMES = (
    "Корреляция + скорейший спуск",
    "Случайный порядок SGD",
    "Порядок по |отступу| SGD",
    "Рекуррентная оценка качества",
    "Эталон: LogisticRegression",
)
metric_models = {}
training_histories = {}
for model_name in METRIC_MODEL_NAMES:
    metric_models[model_name], training_histories[model_name] = fit_metric_model(
        model_name,
        X_train,
        y_train,
        track_loss=model_name == "Эталон: LogisticRegression",
    )

models = {MAIN_MODEL_NAME: best_model, **metric_models}
training_histories = {
    MAIN_MODEL_NAME: primary_loss_history,
    **training_histories,
}

with (MODELS_DIR / "training_loss_history.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as history_file:
    writer = csv.DictWriter(
        history_file,
        fieldnames=("model", "step", "train_half_mse"),
    )
    writer.writeheader()
    for name, history in training_histories.items():
        writer.writerows(
            {
                "model": name,
                "step": step,
                "train_half_mse": loss,
            }
            for step, loss in enumerate(history, start=1)
        )


def model_scores(model, features):
    if isinstance(model, LinearClassifier):
        return model.predict_score(features)
    positive_class_index = list(model.classes_).index(1)
    return 2 * model.predict_proba(features)[:, positive_class_index] - 1


def calculate_metrics(model, train_features, test_features, train_targets, test_targets):
    train_scores = model_scores(model, train_features)
    test_scores = model_scores(model, test_features)
    predictions = np.where(test_scores >= 0, 1, -1)

    return {
        "train_half_mse": float(np.mean((train_targets - train_scores) ** 2) / 2),
        "test_half_mse": float(np.mean((test_targets - test_scores) ** 2) / 2),
        "accuracy": float(accuracy_score(test_targets, predictions)),
        "precision": float(precision_score(test_targets, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(test_targets, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(test_targets, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_targets, test_scores)),
    }


results = {}
for name, model in models.items():
    if isinstance(model, LinearClassifier):
        train_features, test_features = X_train, X_test
    else:
        train_features, test_features = X_train, X_test
    results[name] = calculate_metrics(
        model,
        train_features,
        test_features,
        y_train,
        y_test,
    )

    metrics = results[name]
    print(
        f"{name}: Accuracy={metrics['accuracy']:.3f}, "
        f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}, "
        f"Test half-MSE={metrics['test_half_mse']:.6f}"
    )

with (MODELS_DIR / "comparison_metrics.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as metrics_file:
    fields = ["model", *next(iter(results.values())).keys()]
    writer = csv.DictWriter(metrics_file, fieldnames=fields)
    writer.writeheader()
    for name, metrics in results.items():
        writer.writerow({"model": name, **metrics})

model_filenames = (
    "main_multistart_sgd_momentum_l2",
    "correlation_steepest_descent",
    "random_order_sgd",
    "margin_order_sgd",
    "recursive_quality_sgd",
    "reference_logistic_regression",
)
for filename, model in zip(model_filenames, models.values()):
    joblib.dump(model, MODELS_DIR / f"{filename}.joblib")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_metrics = {
    name: {metric: [] for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    for name in models
}
for fold_number, (fold_train, fold_validation) in enumerate(cv.split(X_train_raw, y_train)):
    scaler = StandardScaler()
    fold_train_scaled = scaler.fit_transform(X_train_raw[fold_train])
    fold_validation_scaled = scaler.transform(X_train_raw[fold_validation])
    fold_train_features = np.column_stack(
        (np.ones(fold_train_scaled.shape[0]), fold_train_scaled)
    )
    fold_validation_features = np.column_stack(
        (np.ones(fold_validation_scaled.shape[0]), fold_validation_scaled)
    )
    fold_targets = y_train[fold_train]
    validation_targets = y_train[fold_validation]

    fold_models = {}
    for name in METRIC_MODEL_NAMES:
        fold_models[name], _ = fit_metric_model(
            name,
            fold_train_features,
            fold_targets,
            random_state=42 + fold_number,
        )

    primary_initial_weights = np.random.default_rng(42 + fold_number).normal(
        0.0,
        0.1,
        size=fold_train_features.shape[1],
    )
    fold_primary = LinearClassifier(
        fold_train_features.shape[1],
        initial_weights=primary_initial_weights,
    )
    fold_primary.fit_sgd_momentum_l2(
        fold_train_features,
        fold_targets,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        l2_lambda=L2_LAMBDA,
        n_iterations=N_ITERATIONS,
        random_state=42 + fold_number,
    )
    fold_models[MAIN_MODEL_NAME] = fold_primary

    for name, model in fold_models.items():
        scores = model_scores(model, fold_validation_features)
        predictions = np.where(scores >= 0, 1, -1)
        cv_metrics[name]["accuracy"].append(accuracy_score(validation_targets, predictions))
        cv_metrics[name]["precision"].append(
            precision_score(validation_targets, predictions, pos_label=1, zero_division=0)
        )
        cv_metrics[name]["recall"].append(
            recall_score(validation_targets, predictions, pos_label=1, zero_division=0)
        )
        cv_metrics[name]["f1"].append(
            f1_score(validation_targets, predictions, pos_label=1, zero_division=0)
        )
        cv_metrics[name]["roc_auc"].append(roc_auc_score(validation_targets, scores))

with (MODELS_DIR / "repeated_cv_metrics.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as cv_file:
    cv_fields = ["model"] + [
        column
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
        for column in (f"{metric}_mean", f"{metric}_std")
    ]
    writer = csv.DictWriter(cv_file, fieldnames=cv_fields)
    writer.writeheader()
    for name, metrics in cv_metrics.items():
        row = {"model": name}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        writer.writerow(row)

metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
model_names = list(models)
positions = np.arange(len(model_names))
bar_width = 0.15
fig, axis = plt.subplots(figsize=(14, 7))
for metric_index, metric_name in enumerate(metric_names):
    values = [results[name][metric_name] for name in model_names]
    axis.bar(
        positions + (metric_index - 2) * bar_width,
        values,
        bar_width,
        label=metric_name.upper(),
    )
axis.set_xticks(positions)
axis.set_xticklabels(model_names, rotation=20, ha="right")
axis.set_ylim(0, 1.05)
axis.set_ylabel("Значение метрики")
axis.set_title("Сравнение исходных вариантов и основной модели")
axis.legend(ncols=3)
axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(GRAPHS_DIR / "metrics_comparison.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for axis, (name, model) in zip(axes.flat, models.items()):
    scores = model_scores(model, X_test)
    predictions = np.where(scores >= 0, 1, -1)
    matrix = confusion_matrix(y_test, predictions, labels=[-1, 1])
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(name, fontsize=9)
    axis.set_xticks([0, 1], labels=["-1", "+1"])
    axis.set_yticks([0, 1], labels=["-1", "+1"])
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
fig.suptitle("Матрицы ошибок")
fig.savefig(GRAPHS_DIR / "confusion_matrices.png", dpi=160, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for axis, metric in zip(axes, ("accuracy", "roc_auc")):
    means = [np.mean(cv_metrics[name][metric]) for name in model_names]
    deviations = [np.std(cv_metrics[name][metric], ddof=1) for name in model_names]
    axis.errorbar(means, np.arange(len(model_names)), xerr=deviations, fmt="o", capsize=4)
    axis.set_yticks(np.arange(len(model_names)), labels=model_names)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel(f"{metric.upper()} (среднее +/- std)")
    axis.grid(axis="x", alpha=0.25)
axes[0].set_title("Stratified 5-fold CV, 3 повтора")
fig.tight_layout()
fig.savefig(GRAPHS_DIR / "cross_validation_comparison.png", dpi=160)
plt.close(fig)

fig, axis = plt.subplots(figsize=(10, 6))
for name, history in training_histories.items():
    if history:
        axis.plot(np.arange(1, len(history) + 1), history, label=name)
axis.set_xlabel("Эпоха / итерация")
axis.set_ylabel("Качество на train")
axis.set_title("Кривые обучения сравниваемых вариантов")
axis.legend()
axis.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(GRAPHS_DIR / "training_curves.png", dpi=160)
plt.close(fig)

fig, axis = plt.subplots(figsize=(8, 5))
axis.plot(np.arange(1, len(multistart_losses) + 1), multistart_losses, marker="o")
axis.set_xlabel("Номер запуска")
axis.set_ylabel("Итоговый train Loss")
axis.set_title("Мультистарт основной модели")
axis.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(GRAPHS_DIR / "multistart_losses.png", dpi=160)
plt.close(fig)

fig, axis = plt.subplots(figsize=(9, 5))
for label, features, targets in (("Train", X_train, y_train), ("Test", X_test, y_test)):
    axis.hist(targets * best_model.predict_score(features), bins=24, alpha=0.55, label=label)
axis.axvline(0, color="black", linewidth=1)
axis.set_xlabel("Отступ y * f(x)")
axis.set_ylabel("Количество объектов")
axis.set_title("Отступы основной модели")
axis.legend()
axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(GRAPHS_DIR / "main_model_margins.png", dpi=160)
plt.close(fig)

fig, axis = plt.subplots(figsize=(9, 5))
quality_history = metric_models["Рекуррентная оценка качества"].quality_history
axis.plot(np.arange(1, len(quality_history) + 1), quality_history)
axis.set_xlabel("Итерация")
axis.set_ylabel("Рекуррентная оценка качества")
axis.set_title("Рекуррентная оценка качества")
axis.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(GRAPHS_DIR / "recursive_quality.png", dpi=160)
plt.close(fig)

print(f"Основная модель и варианты сравнения сохранены в: {MODELS_DIR}")
print(f"Метрики сохранены в: {MODELS_DIR / 'comparison_metrics.csv'}")
print(f"CV-метрики сохранены в: {MODELS_DIR / 'repeated_cv_metrics.csv'}")
print(f"Графики сохранены в: {GRAPHS_DIR}")