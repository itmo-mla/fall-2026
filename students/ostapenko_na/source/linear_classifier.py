from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ITER = 5000
LEARNING_RATE = 0.01
GAMMA = 0.9
LAMBDA_Q = 0.01
TAU = 0.001
Q_INIT_FRACTION = 0.10
N_STARTS = 10

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def load_banknote():
    """Загружает UCI Banknote Authentication."""

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=267)
    X_df = dataset.data.features.copy()
    y = dataset.data.targets.to_numpy().ravel()
    return X_df, y

def prepare_data():
    X_df, y_raw = load_banknote()

    y_numeric = pd.to_numeric(pd.Series(y_raw), errors="raise").to_numpy()
    y = np.where(y_numeric == 0, -1, 1)

    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names



def margin(X, y, w):
    return y * (X @ w)


def loss(m):
    return (1 - m) ** 2


def loss_gradient(x, y, w):
    m = margin(x, y, w)
    return -2 * y * x * (1 - m)


def mean_loss(X, y, w):
    return float(np.mean(loss(margin(X, y, w))))


def predict_scores(X, w):
    return X @ w


def predict(X, w):
    return np.where(predict_scores(X, w) >= 0, 1, -1)

def correlation_init(X, y):
    """w_j = <y, f_j> / <f_j, f_j>."""
    numerator = X.T @ y
    denominator = np.sum(X ** 2, axis=0)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return numerator / denominator


def random_init(n_features, rng):
    """w_j ~ U(-1/(2n), 1/(2n))."""
    bound = 1 / (2 * n_features)
    return rng.uniform(-bound, bound, size=n_features)


def initialize_q(X, y, w, rng, fraction=Q_INIT_FRACTION):
    subset_size = max(1, int(fraction * len(X)))
    idx = rng.choice(len(X), size=subset_size, replace=False)
    return mean_loss(X[idx], y[idx], w)


def choose_object(X, y, w, rng, sampling="random", eps=1e-8):
    if sampling == "random":
        return int(rng.integers(len(X)))

    if sampling == "margin":
        m = margin(X, y, w)
        weights = 1.0 / (np.abs(m) + eps)
        probabilities = weights / weights.sum()
        return int(rng.choice(len(X), p=probabilities))

    raise ValueError("sampling должен быть 'random' или 'margin'")


# ---------------------------------------------------------------------
# SGD + momentum + L2
# ---------------------------------------------------------------------

def fit_sgd(
    X, y, w_init,
    n_iter=N_ITER,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    lambda_q=LAMBDA_Q,
    tau=TAU,
    sampling="random",
    random_state=RANDOM_STATE
):
    rng = np.random.default_rng(random_state)

    w = np.asarray(w_init, dtype=float).copy()
    v = np.zeros_like(w)

    Q = initialize_q(X, y, w, rng)
    q_history = [Q]

    eval_every = max(1, n_iter // 200)
    train_loss_history = [(0, mean_loss(X, y, w))]

    for iteration in range(1, n_iter + 1):
        i = choose_object(X, y, w, rng, sampling)

        x_i = X[i]
        y_i = y[i]

        m_i = margin(x_i, y_i, w)
        loss_i = loss(m_i)

        # Градиент квадратичной функции потерь
        grad = loss_gradient(x_i, y_i, w)

        # Momentum
        v = gamma * v + (1 - gamma) * grad

        # L2-регуляризация по формуле из лекции:
        w = (
            (1 - learning_rate * tau) * w
            - learning_rate * v
        )

        # Рекуррентная оценка функционала качества
        Q = (
            lambda_q * loss_i
            + (1 - lambda_q) * Q
        )

        q_history.append(float(Q))

        if iteration % eval_every == 0 or iteration == n_iter:
            train_loss_history.append(
                (iteration, mean_loss(X, y, w))
            )

    return {
        "w": w,
        "q_history": np.asarray(q_history),
        "train_loss_history": np.asarray(
            train_loss_history,
            dtype=float
        ),
    }


# ---------------------------------------------------------------------
# Скорейший градиентный спуск
# ---------------------------------------------------------------------

def fit_steepest(
    X,
    y,
    w_init,
    n_iter=N_ITER,
    lambda_q=LAMBDA_Q,
    random_state=RANDOM_STATE,
    eps=1e-12
):
    rng = np.random.default_rng(random_state)

    w = np.asarray(w_init, dtype=float).copy()

    Q = initialize_q(X, y, w, rng)
    q_history = [Q]

    eval_every = max(1, n_iter // 200)

    train_loss_history = [
        (0, mean_loss(X, y, w))
    ]

    for iteration in range(1, n_iter + 1):

        i = int(rng.integers(len(X)))

        x_i = X[i]
        y_i = y[i]

        m_i = margin(x_i, y_i, w)
        loss_i = loss(m_i)

        grad = loss_gradient(
            x_i,
            y_i,
            w
        )
        
        h_star = 1.0 / (
            2.0 * np.dot(x_i, x_i) + eps
        )
        w = w - h_star * grad

        Q = (
            lambda_q * loss_i
            + (1 - lambda_q) * Q
        )

        q_history.append(float(Q))

        if (
            iteration % eval_every == 0
            or iteration == n_iter
        ):
            train_loss_history.append(
                (
                    iteration,
                    mean_loss(X, y, w)
                )
            )

    return {
        "w": w,
        "q_history": np.asarray(q_history),
        "train_loss_history": np.asarray(
            train_loss_history,
            dtype=float
        ),
    }

# ---------------------------------------------------------------------
# Multistart
# ---------------------------------------------------------------------

def fit_multistart(X, y, n_starts=N_STARTS):
    rng = np.random.default_rng(RANDOM_STATE)
    best_result = None
    best_loss = np.inf
    rows = []

    for start in range(1, n_starts + 1):
        w_init = random_init(X.shape[1], rng)
        seed = int(rng.integers(0, 2**32 - 1))

        result = fit_sgd(
            X, y,
            w_init=w_init,
            sampling="random",
            random_state=seed
        )
        current_loss = mean_loss(X, y, result["w"])
        rows.append({"start": start, "train_loss": current_loss})

        if current_loss < best_loss:
            best_loss = current_loss
            best_result = result

    return best_result, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------

def metrics_for_custom(name, X_train, y_train, X_test, y_test, w):
    train_pred = predict(X_train, w)
    test_pred = predict(X_test, w)
    scores = predict_scores(X_test, w)

    return {
        "model": name,
        "train_loss": mean_loss(X_train, y_train, w),
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
        "train_loss": np.nan,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, pos_label=1),
        "recall": recall_score(y_test, test_pred, pos_label=1),
        "f1": f1_score(y_test, test_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, scores),
    }


def save_q_plot(histories):
    plt.figure(figsize=(10, 5))
    for label, h in histories.items():
        plt.plot(h, label=label)
    plt.xlabel("Итерация")
    plt.ylabel("Q")
    plt.title("Рекуррентная оценка функционала качества")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "q_history.png", dpi=160)
    plt.close()


def save_train_loss_plot(histories):
    plt.figure(figsize=(10, 5))
    for label, h in histories.items():
        plt.plot(h[:, 0], h[:, 1], label=label)
    plt.xlabel("Итерация")
    plt.ylabel("Средний loss на train")
    plt.title("Динамика среднего train loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "train_loss.png", dpi=160)
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
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "metrics_comparison.png", dpi=160)
    plt.close()


def save_roc_plot(curves):
    plt.figure(figsize=(8, 6))
    for label, (y_true, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{label}, AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Случайный классификатор")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "roc_curves.png", dpi=160)
    plt.close()


def save_margin_plot(X, y, w):
    m = np.sort(margin(X, y, w))
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(m)), m, label="Отступы объектов")
    plt.axhline(0, linestyle="--", label="Граница M = 0")
    plt.xlabel("Объекты, отсортированные по отступу")
    plt.ylabel("Отступ M")
    plt.title("Распределение отступов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "margins.png", dpi=160)
    plt.close()


def save_multistart_plot(starts):
    best_idx = starts["train_loss"].idxmin()
    plt.figure(figsize=(8, 5))
    plt.bar(
        starts["start"].astype(str),
        starts["train_loss"],
        label="Train loss каждого старта"
    )
    plt.axhline(
        starts.loc[best_idx, "train_loss"],
        linestyle="--",
        label=f"Лучший loss = {starts.loc[best_idx, 'train_loss']:.4f}"
    )
    plt.xlabel("Номер старта")
    plt.ylabel("Train loss")
    plt.title("Сравнение случайных инициализаций multistart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "multistart.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# Основной эксперимент
# ---------------------------------------------------------------------

def main():
    X_train, X_test, y_train, y_test, feature_names = prepare_data()

    print("Признаки:", feature_names)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    classes, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(classes, counts):
        print(
            f"Класс {int(cls):+d}: {count} объектов "
            f"({count / len(y_train):.1%})"
        )
    w_corr = correlation_init(X_train, y_train)

    # 1. Корреляционная инициализация + random sampling.
    corr_random = fit_sgd(
        X_train, y_train,
        w_init=w_corr,
        sampling="random",
        random_state=RANDOM_STATE
    )

    # 2. Random init + multistart.
    multistart, starts = fit_multistart(X_train, y_train)

    # 3. Корреляционная инициализация + sampling по |margin|.
    corr_margin = fit_sgd(
        X_train, y_train,
        w_init=w_corr,
        sampling="margin",
        random_state=RANDOM_STATE
    )

    # 4. Скорейший градиентный спуск.
    steepest = fit_steepest(
        X_train, y_train,
        w_init=w_corr,
        random_state=RANDOM_STATE
    )

    # 5. Эталонная библиотечная реализация.
    reference = SGDClassifier(
        loss="squared_error",
        penalty="l2",
        alpha=TAU,
        fit_intercept=False,
        max_iter=N_ITER,
        tol=1e-6,
        random_state=RANDOM_STATE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference.fit(X_train, y_train)

    custom = {
        "Correlation + random": corr_random,
        "Random + multistart": multistart,
        "Correlation + |margin|": corr_margin,
        "Steepest gradient": steepest,
    }

    rows = [
        metrics_for_custom(
            name, X_train, y_train, X_test, y_test, result["w"]
        )
        for name, result in custom.items()
    ]
    rows.append(
        metrics_for_reference(
            "sklearn SGDClassifier",
            reference, X_train, y_train, X_test, y_test
        )
    )
    results = pd.DataFrame(rows)

    print("\nИтоговые метрики:")
    print(results.round(4).to_string(index=False))
    print("\nMultistart:")
    print(starts.round(5).to_string(index=False))

    # Графики.
    save_q_plot({
        name: result["q_history"]
        for name, result in custom.items()
    })
    save_train_loss_plot({
        name: result["train_loss_history"]
        for name, result in custom.items()
    })
    save_metrics_plot(results)

    curves = {
        name: (y_test, predict_scores(X_test, result["w"]))
        for name, result in custom.items()
    }
    curves["sklearn SGDClassifier"] = (
        y_test, reference.decision_function(X_test)
    )
    save_roc_plot(curves)
    save_margin_plot(X_train, y_train, corr_random["w"])
    save_multistart_plot(starts)

    print(f"\nГрафики сохранены в: {IMAGE_DIR}")
    results.to_csv(
        Path(__file__).resolve().parent / "metrics.csv",
        index=False
    )

    starts.to_csv(
        Path(__file__).resolve().parent / "multistart_results.csv",
        index=False
)

if __name__ == "__main__":
    main()
