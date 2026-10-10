from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_and_preprocess_data(test_size: float = 0.20, random_state: int = 42):
    data = load_breast_cancer()
    X = data.data
    y = np.where(data.target == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, list(data.feature_names)


# -------------------------------------------------------------
# Метрики (Шаг 10 и 11)
# -------------------------------------------------------------
def metrics_for_custom(name, X_train, y_train, X_test, y_test, w, mean_loss_fn, predict_fn, predict_scores_fn):
    train_pred = predict_fn(X_train, w)
    test_pred = predict_fn(X_test, w)
    scores = predict_scores_fn(X_test, w)

    return {
        "model": name,
        "train_loss": mean_loss_fn(X_train, y_train, w),
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, pos_label=1),
        "recall": recall_score(y_test, test_pred, pos_label=1),
        "f1": f1_score(y_test, test_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, scores),
    }


def metrics_for_reference(name, model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    scores = model.decision_function(X_test)

    return {
        "model": name,
        "train_loss": float(np.mean((y_train - model.decision_function(X_train)) ** 2)),
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, pos_label=1),
        "recall": recall_score(y_test, test_pred, pos_label=1),
        "f1": f1_score(y_test, test_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, scores),
    }


# -------------------------------------------------------------
# Построение графиков
# -------------------------------------------------------------
def save_q_plot(histories):
    plt.figure(figsize=(10, 5))
    for label, h in histories.items():
        plt.plot(h, label=label)
    plt.xlabel("Итерация")
    plt.ylabel("Q")
    plt.title("Рекуррентная оценка функционала качества")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "q_history.png", dpi=200)
    plt.close()


def save_train_loss_plot(histories):
    plt.figure(figsize=(10, 5))
    for label, h in histories.items():
        plt.plot(h[:, 0], h[:, 1], label=label)
    plt.xlabel("Итерация")
    plt.ylabel("Средний loss на train")
    plt.title("Динамика среднего train loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "train_loss.png", dpi=200)
    plt.close()


def save_metrics_plot(df):
    metrics = ["test_accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = df.set_index("model")[metrics].T.plot(kind="bar", figsize=(12, 6))
    ax.set_xlabel("Метрика")
    ax.set_ylabel("Значение")
    ax.set_title("Сравнение качества моделей на test")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Модель", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "metrics_comparison.png", dpi=200)
    plt.close()


def save_roc_plot(curves):
    plt.figure(figsize=(8, 6))
    for label, (y_true, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{label}, AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Случайный классификатор")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "roc_curves.png", dpi=200)
    plt.close()


def save_margin_plot(X, y, w, margin_fn):
    m = np.sort(margin_fn(X, y, w))
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(m)), m, label="Отступы объектов M_i", color="blue")
    plt.axhline(0, color="red", linestyle="--", label="Граница M = 0")
    plt.xlabel("Объекты, отсортированные по отступу")
    plt.ylabel("Отступ M")
    plt.title("Распределение отступов")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "margins.png", dpi=200)
    plt.close()


def save_multistart_plot(starts):
    best_idx = starts["train_loss"].idxmin()
    plt.figure(figsize=(8, 5))
    plt.bar(starts["start"].astype(str), starts["train_loss"], label="Train loss каждого старта", color="teal")
    plt.axhline(
        starts.loc[best_idx, "train_loss"],
        color="red",
        linestyle="--",
        label=f"Лучший loss = {starts.loc[best_idx, 'train_loss']:.4f}",
    )
    plt.xlabel("Номер старта")
    plt.ylabel("Train loss")
    plt.title("Сравнение случайных инициализаций multistart")
    plt.legend(loc="lower left")
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "multistart.png", dpi=200)
    plt.close()
