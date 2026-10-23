from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression


SEED = 42
TEST_SIZE = 0.20
L2 = 0.01
MAX_ITER = 100
TOL = 1e-10
THRESHOLD = 0.5
PLOTS_DIR = Path("plots")
CLASS_NAMES = ("B (доброкачественная)", "M (злокачественная)")


def load_data():
    dataset = fetch_ucirepo(id=17)
    features = dataset.data.features
    targets = dataset.data.targets.iloc[:, 0].astype(str).str.strip()

    if set(targets.unique()) != {"B", "M"}:
        raise ValueError("Ожидались две метки Diagnosis: 'B' и 'M'.")

    x = features.to_numpy(dtype=float)
    y = (targets.to_numpy() == "M").astype(int)
    if not np.isfinite(x).all():
        raise ValueError("Обнаружены пропуски или некорректные числовые признаки.")
    return x, y, features.columns.to_list()


def stratified_split(y, test_size=TEST_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    n_test_total = int(np.ceil(len(y) * test_size))
    if not (0 < n_test_total < len(y)):
        raise ValueError("Некорректная доля test_size.")
    quotas = counts * n_test_total / len(y)
    test_counts = np.floor(quotas).astype(int)
    remainder = n_test_total - int(test_counts.sum())
    for i in np.argsort(-(quotas - test_counts), kind="stable")[:remainder]:
        test_counts[i] += 1
    if np.any(test_counts < 1) or np.any(test_counts >= counts):
        raise ValueError("Для стратификации нужны оба класса в train и test.")
    train_parts, test_parts = [], []
    for label, n_test in zip(labels, test_counts):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        test_parts.append(indices[:n_test])
        train_parts.append(indices[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def standardize_train_test(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    scale = np.sqrt(np.mean((x_train - mean) ** 2, axis=0))
    scale[scale == 0] = 1.0
    return (x_train - mean) / scale, (x_test - mean) / scale


def add_intercept(x):
    return np.column_stack((np.ones(x.shape[0]), x))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def reg_mask(n_params):
    mask = np.ones(n_params)
    mask[0] = 0.0
    return mask


def objective(x, y, w, l2=L2):
    logits = x @ w
    data_loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
    return float(data_loss + 0.5 * l2 * np.dot(w[1:], w[1:]))


def gradient(x, y, w, l2=L2):
    p = sigmoid(x @ w)
    mask = reg_mask(x.shape[1])
    return x.T @ (p - y) / len(y) + l2 * (mask * w)


def hessian(x, w, l2=L2):
    p = sigmoid(x @ w)
    weights = np.maximum(p * (1.0 - p), 1e-12)
    return x.T @ (weights[:, None] * x) / x.shape[0] + l2 * np.diag(reg_mask(x.shape[1]))


def backtracking(x, y, w, candidate, l2=L2):
    current = objective(x, y, w, l2)
    alpha = 1.0
    for _ in range(35):
        proposal = w + alpha * (candidate - w)
        if objective(x, y, proposal, l2) <= current + 1e-14:
            return proposal
        alpha *= 0.5
    return w.copy()


def new_history(x, y, w, l2):
    return {"loss": [objective(x, y, w, l2)],
            "grad_norm": [float(np.linalg.norm(gradient(x, y, w, l2)))]}


def record_step(history, x, y, w, l2):
    history["loss"].append(objective(x, y, w, l2))
    history["grad_norm"].append(float(np.linalg.norm(gradient(x, y, w, l2))))


def converged(previous, current, grad_norm, tol=TOL):
    step_norm = np.linalg.norm(current - previous)
    return grad_norm < tol or step_norm < tol * (1.0 + np.linalg.norm(current))


def fit_newton(x, y, l2=L2, max_iter=MAX_ITER, tol=TOL):
    w = np.zeros(x.shape[1], dtype=float)
    history = new_history(x, y, w, l2)
    for _ in range(max_iter):
        g = gradient(x, y, w, l2)
        if np.linalg.norm(g) < tol:
            break
        delta = np.linalg.solve(hessian(x, w, l2), g)
        updated = backtracking(x, y, w, w - delta, l2)
        record_step(history, x, y, updated, l2)
        done = converged(w, updated, history["grad_norm"][-1], tol)
        w = updated
        if done:
            break
    return w, history


def fit_irls(x, y, l2=L2, max_iter=MAX_ITER, tol=TOL):


    w = np.zeros(x.shape[1], dtype=float)
    history = new_history(x, y, w, l2)
    n = x.shape[0]
    penalty = l2 * np.diag(reg_mask(x.shape[1]))
    for _ in range(max_iter):
        if np.linalg.norm(gradient(x, y, w, l2)) < tol:
            break
        p = sigmoid(x @ w)
        weights = np.maximum(p * (1.0 - p), 1e-12)
        working_response = x @ w + (y - p) / weights
        lhs = x.T @ (weights[:, None] * x) / n + penalty
        rhs = x.T @ (weights * working_response) / n
        candidate = np.linalg.solve(lhs, rhs)
        updated = backtracking(x, y, w, candidate, l2)
        record_step(history, x, y, updated, l2)
        done = converged(w, updated, history["grad_norm"][-1], tol)
        w = updated
        if done:
            break
    return w, history


def fit_reference(x_train_scaled, y_train, l2=L2):


    if l2 <= 0:
        raise ValueError("Для эталонной модели в этом эксперименте l2 должен быть > 0.")
    model = LogisticRegression(
        solver="lbfgs", C=1.0 / (len(y_train) * l2),
        fit_intercept=True, tol=1e-12, max_iter=5000, random_state=SEED,
    )
    model.fit(x_train_scaled, y_train)
    return np.r_[model.intercept_, model.coef_.ravel()]


def predict_proba(x, w):
    return sigmoid(x @ w)


def predict_labels(probabilities, threshold=THRESHOLD):
    return (probabilities >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def roc_curve_and_auc(y_true, probabilities):
    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC требует наличия обоих классов.")
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_y = y_true[order]
    sorted_prob = probabilities[order]
    last_of_ties = np.r_[np.flatnonzero(np.diff(sorted_prob) != 0), len(sorted_prob) - 1]
    tp = np.cumsum(sorted_y)[last_of_ties]
    fp = (1 + last_of_ties) - tp
    tpr = np.r_[0.0, tp / n_positive]
    fpr = np.r_[0.0, fp / n_negative]

    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))
    return fpr, tpr, auc


def evaluate(y_true, probabilities):
    predictions = predict_labels(probabilities)
    matrix = confusion_counts(y_true, predictions)
    tn, fp, fn, tp = matrix.ravel()
    n = len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_safe = np.clip(probabilities, 1e-15, 1 - 1e-15)
    log_loss = float(-np.mean(y_true * np.log(p_safe) + (1 - y_true) * np.log(1 - p_safe)))
    fpr, tpr, auc = roc_curve_and_auc(y_true, probabilities)
    return {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "log_loss": log_loss,
        "auc": auc,
        "confusion": matrix,
        "fpr": fpr,
        "tpr": tpr,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def save_plot(fig, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(y_train, y_test):
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = (np.r_[y_train, y_test], y_train, y_test)
    x = np.arange(len(datasets))
    benign = [np.sum(values == 0) for values in datasets]
    malignant = [np.sum(values == 1) for values in datasets]
    ax.bar(x, benign, label="B: доброкачественная")
    ax.bar(x, malignant, bottom=benign, label="M: злокачественная")
    ax.set_xticks(x, ("Вся выборка", "Train", "Test"))
    ax.set_ylabel("Количество объектов")
    ax.set_title("Распределение классов при стратифицированном разделении")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "01_class_balance.png")


def plot_convergence(history_newton, history_irls):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, history, marker in (
        ("Ньютон–Рафсон", history_newton, "o"),
        ("IRLS", history_irls, "x"),
    ):
        steps = np.arange(len(history["loss"]))
        axes[0].plot(steps, history["loss"], marker=marker, label=name)
        axes[1].semilogy(steps, np.maximum(history["grad_norm"], 1e-16),
                        marker=marker, label=name)
    axes[0].set_title("Сходимость функции потерь")
    axes[0].set_ylabel("Средняя NLL + L2")
    axes[1].set_title("Уменьшение нормы градиента")
    axes[1].set_ylabel("||grad L||, логарифмическая шкала")
    for ax in axes:
        ax.set_xlabel("Номер итерации (0 — начальное состояние)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    save_plot(fig, "02_convergence.png")


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, info in results.items():
        ax.plot(info["fpr"], info["tpr"], lw=2,
                label=f"{name}: AUC = {info['auc']:.5f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Случайный классификатор (AUC = 0.5)")
    ax.set(xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)",
           title="ROC-кривые на тестовой выборке", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    save_plot(fig, "03_roc.png")


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.5))
    max_count = max(int(info["confusion"].max()) for info in results.values())
    im = None
    cell_names = (("TN", "FP"), ("FN", "TP"))
    for ax, (name, info) in zip(axes, results.items()):
        cm = info["confusion"]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max_count)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{cell_names[row][col]}\n{cm[row, col]}",
                        ha="center", va="center", fontsize=12,
                        color="white" if cm[row, col] > max_count / 2 else "black")
        ax.set_xticks((0, 1), ("B (0)", "M (1)"))
        ax.set_yticks((0, 1), ("B (0)", "M (1)"))
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title(name)

    colorbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Количество объектов")
    fig.suptitle("Матрицы ошибок (положительный класс — M)")
    fig.subplots_adjust(top=0.82, wspace=0.5, right=0.88)
    save_plot(fig, "04_confusion_matrices.png")


def plot_coefficients(feature_names, weights):
    order = np.argsort(np.abs(weights["Ньютон–Рафсон"][1:]))
    positions = np.arange(len(feature_names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 12))
    for offset, (name, w) in zip((-width, 0.0, width), weights.items()):
        ax.barh(positions + offset, w[1:][order], height=width, label=name)
    ax.set_yticks(positions, [feature_names[i] for i in order])
    ax.set_xlabel("Вес стандартизированного признака")
    ax.set_title("Сравнение коэффициентов трёх моделей (без intercept)")
    ax.axvline(0, linestyle="--", linewidth=1, label="Нулевой коэффициент")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    save_plot(fig, "05_coefficients.png")


def plot_probability_agreement(results):
    reference = results["sklearn"]["probabilities"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in ("Ньютон–Рафсон", "IRLS"):
        values = results[name]["probabilities"]
        ax.scatter(reference, values, alpha=0.65, s=24,
                   label=f"{name}; max |Δp|={np.max(np.abs(values-reference)):.2e}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Идеальное совпадение: y = x")
    ax.set(xlabel="Вероятность M: sklearn", ylabel="Вероятность M: своя модель",
           title="Эквивалентность предсказанных вероятностей", xlim=(-0.02, 1.02),
           ylim=(-0.02, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    save_plot(fig, "06_probability_agreement.png")


def print_results(results, weights, histories, x_train, y_train, l2):
    print("\nМетрики на ОДНОЙ тестовой выборке (M = положительный класс):")
    header = f"{'Метод':<19} {'Accuracy':>9} {'Precision':>10} {'Recall':>9} {'Specificity':>12} {'F1':>9} {'ROC AUC':>9} {'Log-loss':>10}"
    print(header)
    print("-" * len(header))
    for name, item in results.items():
        print(f"{name:<19} {item['accuracy']:>9.5f} {item['precision']:>10.5f} "
              f"{item['recall']:>9.5f} {item['specificity']:>12.5f} "
              f"{item['f1']:>9.5f} {item['auc']:>9.5f} {item['log_loss']:>10.5f}")

    print("\nСходимость и проверка эквивалентности:")
    for name, w in weights.items():
        print(f"{name:<19}: L_train={objective(x_train, y_train, w, l2):.12f}, "
              f"||grad||={np.linalg.norm(gradient(x_train, y_train, w, l2)):.3e}")
    for name, hist in histories.items():
        print(f"Итераций {name}: {len(hist['loss'])-1}")
    w_newton, w_irls, w_ref = weights.values()
    print(f"max |w_Newton - w_IRLS|       = {np.max(np.abs(w_newton-w_irls)):.3e}")
    print(f"max |w_Newton - w_sklearn|    = {np.max(np.abs(w_newton-w_ref)):.3e}")
    print(f"max |w_IRLS - w_sklearn|      = {np.max(np.abs(w_irls-w_ref)):.3e}")
    for name in ("Ньютон–Рафсон", "IRLS"):
        ref = results["sklearn"]
        current = results[name]
        print(f"{name}: max |p - p_sklearn| = "
              f"{np.max(np.abs(current['probabilities'] - ref['probabilities'])):.3e}, "
              f"несовпадений классов = {np.sum(current['predictions'] != ref['predictions'])}")
    np.testing.assert_allclose(w_newton, w_irls, rtol=1e-6, atol=1e-7,
                               err_msg="Ньютон и IRLS должны давать одинаковое решение")
    print("Проверка Newton ≈ IRLS: ПРОЙДЕНА")


def main():
    x, y, feature_names = load_data()
    train_idx, test_idx = stratified_split(y)
    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train_scaled, x_test_scaled = standardize_train_test(x_train_raw, x_test_raw)
    x_train = add_intercept(x_train_scaled)
    x_test = add_intercept(x_test_scaled)

    print("Датасет: UCI Breast Cancer Wisconsin (Diagnostic), ID=17")
    print(f"Объектов: {len(y)}; признаков: {x.shape[1]}; train: {len(y_train)}; test: {len(y_test)}")
    print(f"Классы: B=0 ({np.sum(y == 0)}), M=1 ({np.sum(y == 1)})")
    print(f"Параметры: seed={SEED}, L2={L2}, test_size={TEST_SIZE}, threshold={THRESHOLD}")

    w_newton, h_newton = fit_newton(x_train, y_train)
    w_irls, h_irls = fit_irls(x_train, y_train)
    w_sklearn = fit_reference(x_train_scaled, y_train)
    weights = {
        "Ньютон–Рафсон": w_newton,
        "IRLS": w_irls,
        "sklearn": w_sklearn,
    }
    histories = {"Ньютон–Рафсон": h_newton, "IRLS": h_irls}
    results = {name: evaluate(y_test, predict_proba(x_test, w))
               for name, w in weights.items()}
    print_results(results, weights, histories, x_train, y_train, L2)

    plot_class_balance(y_train, y_test)
    plot_convergence(h_newton, h_irls)
    plot_roc(results)
    plot_confusion_matrices(results)
    plot_coefficients(feature_names, weights)
    plot_probability_agreement(results)



if __name__ == "__main__":
    main()
