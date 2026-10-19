import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from data import (
    CLASS_NAMES,
    PLOT_NAMES,
    PLOTS_DIR,
    as_numpy,
    load_raw_dataframe,
    prepare_data,
    visual_arrays,
)
from synthetic import prepare_circles

warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

C_SOFT_MARGIN = 1.0
SUPPORT_TOL = 1e-6

#Линейное ядро
def linear_kernel(X, Z):
    return np.asarray(X, dtype=float) @ np.asarray(Z, dtype=float).T

#  Гауссово ядро
def rbf_kernel(X, Z, gamma):
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    xx = np.sum(X * X, axis=1)[:, None]
    zz = np.sum(Z * Z, axis=1)[None, :]
    distance_sq = np.maximum(xx + zz - 2.0 * X @ Z.T, 0.0)
    return np.exp(-gamma * distance_sq)


def gamma_after_scaling(X):
    X = np.asarray(X, dtype=float)
    return 1.0 / (X.shape[1] * float(X.var()))


# Для RBF γ пришит к функции заранее, чтобы fit_dual не зависел от вида ядра.
def make_kernel(kernel_name, gamma=None):
    if kernel_name == "linear":
        return linear_kernel
    if kernel_name == "rbf":
        if gamma is None:
            raise ValueError("Для RBF нужен gamma, тот же, что будет передан в SVC.")

        def kernel(X, Z, gamma=gamma):
            return rbf_kernel(X, Z, gamma)

        return kernel
    raise ValueError(f"Неизвестное ядро: {kernel_name}")

#для minimize
def _quadratic(alpha, Q):
    return 0.5 * alpha @ Q @ alpha - alpha.sum()

# Двойственная цель на конкретном лямба
# Q_ij = y_i y_j K(x_i, x_j). Метки обязаны быть ±1: при метках 0/1 матрица Q теряет знак класса.
def dual_quadratic(X, y, alpha, kernel=linear_kernel):
    y = np.asarray(y, dtype=float)
    K = kernel(X, X)
    Q = np.outer(y, y) * K
    return _quadratic(alpha, Q)

# Градиент 0.5 λ^T Q λ − 1^T λ.
# Q симметрична, поэтому градиент равен Q λ − 1.
def dual_gradient(alpha, Q):
    return Q @ alpha - np.ones_like(alpha)

#Решение двойственной задачи по лямбде
# Ищем лмябда который минимизирует градиент ф-и
def fit_dual(X, y, C=C_SOFT_MARGIN, kernel=linear_kernel):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples = X.shape[0]
    K = kernel(X, X)
    Q = np.outer(y, y) * K

    def objective(alpha):
        return _quadratic(alpha, Q)

    def gradient(alpha):
        return dual_gradient(alpha, Q)
    constraint = {
        "type": "eq",
        "fun": lambda alpha: alpha @ y,
        "jac": lambda alpha: y,
    }
    bounds = [(0.0, C) for _ in range(n_samples)]
    result = minimize(
        fun=objective,
        x0=np.zeros(n_samples),
        jac=gradient,
        bounds=bounds,
        constraints=constraint,
        method="SLSQP",
        options={
            "ftol": 1e-9,
            "maxiter": 2000,
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(result.message)
    alpha = np.asarray(result.x, dtype=float)
    if not np.isfinite(alpha).all():
        raise RuntimeError("SLSQP вернул нечисловые λ.")
    return alpha, K, result


def calculate_bias(alpha, X, y, K, C, tol=SUPPORT_TOL):
    if K.shape[0] != len(np.asarray(X)):
        raise ValueError("Число строк K и число объектов X не совпали.")
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    margin_indices = np.flatnonzero((alpha > tol) & (alpha < C - tol))
    if len(margin_indices) == 0:
        raise RuntimeError(
            "Нет свободных опорных векторов 0 < λ < C. "
            "По точкам с λ = C сдвиг b считать нельзя. Возьмите другое C."
        )
    bias_values = []
    for i in margin_indices:
        score_without_bias = np.sum(alpha * y * K[:, i])
        bias_values.append(y[i] - score_without_bias)
    return float(np.mean(bias_values))


def fit_model(X, y, C=C_SOFT_MARGIN, kernel_name="linear", gamma=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    kernel = make_kernel(kernel_name, gamma=gamma)
    alpha, K, result = fit_dual(X, y, C=C, kernel=kernel)
    alpha = np.clip(alpha, 0.0, C)
    alpha[alpha < SUPPORT_TOL] = 0.0
    b = calculate_bias(alpha, X, y, K, C)
    return {
        "X_train": X,
        "y_train": y,
        "alpha": alpha,
        "b": b,
        "kernel": kernel,
        "kernel_name": kernel_name,
        "gamma": gamma,
        "C": C,
        "result": result,
    }


def decision_function(model, X):
    K_test = model["kernel"](X, model["X_train"])
    return K_test @ (model["alpha"] * model["y_train"]) + model["b"]


def predict(model, X):
    scores = decision_function(model, X)
    return np.where(scores >= 0.0, 1, -1).astype(int)


def linear_weights(model):
    if model["kernel_name"] != "linear":
        raise RuntimeError("Вектор w есть только у линейного ядра.")
    return model["X_train"].T @ (model["alpha"] * model["y_train"])


def print_linear_classifier(model, feature_names):
    w = linear_weights(model)
    scores_from_weights = model["X_train"] @ w + model["b"]
    scores_from_kernel = decision_function(model, model["X_train"])
    gap = float(np.max(np.abs(scores_from_weights - scores_from_kernel)))
    print("линейный классификатор f(x) = w·x + b")
    for name, weight in zip(feature_names, w):
        print(f"  w[{name}] = {weight:.4f}")
    print(f"  b = {model['b']:.4f}")
    print(f"  ширина зазора 1/||w|| = {1.0 / np.linalg.norm(w):.4f}")
    print(f"  max |w·x+b − ядерная форма| на train = {gap:.3e}")


def print_solver_status(model):
    result = model["result"]
    alpha = model["alpha"]
    y = model["y_train"]
    C = model["C"]
    scores = decision_function(model, model["X_train"])
    margin = y * scores
    free = (alpha > SUPPORT_TOL) & (alpha < C - SUPPORT_TOL)
    on_box = alpha >= C - SUPPORT_TOL
    print(
        f"SLSQP: {result.message}, итераций {result.nit}, "
        f"минимум 0.5 λ^T Q λ − 1^T λ = {result.fun:.4f}"
    )
    print(f"ограничение y·λ = {float(alpha @ y):.3e}")
    print(
        f"опорных векторов: {int(np.sum(alpha > SUPPORT_TOL))}, "
        f"свободных (0 < λ < C): {int(np.sum(free))}, "
        f"на потолке λ = C: {int(np.sum(on_box))}"
    )
    if np.any(free):
        deviation = np.abs(margin[free] - 1.0)
        print(
            f"|y f(x) − 1| на свободных опорных: "
            f"среднее {float(deviation.mean()):.3e}, макс {float(deviation.max()):.3e}"
        )


def _scores(y_true, y_pred, scores):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "cm": confusion_matrix(y_true, y_pred, labels=[-1, 1]),
    }


def print_split_metrics(split_name, y_true, model, X):
    y_pred = predict(model, X)
    scores = decision_function(model, X)
    metrics = _scores(y_true, y_pred, scores)
    print(
        f"{split_name}: accuracy={metrics['accuracy']:.4f} "
        f"precision+1={metrics['precision']:.4f} "
        f"recall+1={metrics['recall']:.4f} "
        f"f1+1={metrics['f1']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )
    print("  матрица ошибок, строки — истина, столбцы — прогноз, порядок [-1, +1]:")
    print(f"  {metrics['cm'][0].tolist()}")
    print(f"  {metrics['cm'][1].tolist()}")
    return metrics


def compare_with_svc(model, X_train, y_train, X_test, y_test):
    params = {"kernel": model["kernel_name"], "C": model["C"]}
    if model["kernel_name"] == "rbf":
        params["gamma"] = model["gamma"]
    reference = SVC(**params)
    reference.fit(X_train, y_train)
    own_pred = predict(model, X_test)
    ref_pred = reference.predict(X_test).astype(int)
    own_scores = decision_function(model, X_test)
    ref_scores = reference.decision_function(X_test)
    agreement = float(np.mean(own_pred == ref_pred))
    score_gap = float(np.max(np.abs(own_scores - ref_scores)))
    ref_metrics = _scores(y_test, ref_pred, ref_scores)
    print(
        f"SVC kernel={model['kernel_name']} C={model['C']} "
        f"gamma={model['gamma']}"
    )
    print(
        f"test SVC: accuracy={ref_metrics['accuracy']:.4f} "
        f"precision+1={ref_metrics['precision']:.4f} "
        f"recall+1={ref_metrics['recall']:.4f} "
        f"f1+1={ref_metrics['f1']:.4f} "
        f"roc_auc={ref_metrics['roc_auc']:.4f}"
    )
    print(
        f"совпадение меток на test: {agreement:.4f}, "
        f"max |f_своя − f_SVC| = {score_gap:.4f}, "
        f"b своя = {model['b']:.4f}, b SVC = {float(reference.intercept_[0]):.4f}, "
        f"опорных у SVC: {int(np.sum(reference.n_support_))}"
    )
    return agreement, score_gap


def plot_solution(model, X, y, path, title, xlabel, ylabel, class_names):
    if X.shape[1] != 2:
        raise ValueError("Границу можно нарисовать только для модели с двумя признаками.")
    if len(X) != len(model["alpha"]):
        raise ValueError("Точки на рисунке должны быть обучающими точками этой модели.")
    pad = 0.8
    x_min, x_max = float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad
    y_min, y_max = float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    scores = decision_function(model, grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    fill = ax.contourf(xx, yy, scores, levels=20, cmap="coolwarm", alpha=0.35)
    ax.contour(xx, yy, scores, levels=[-1.0, 1.0], colors="black", linestyles="--", linewidths=1)
    ax.contour(xx, yy, scores, levels=[0.0], colors="black", linewidths=1.6)
    colors = {-1: "#4C72B0", 1: "#DD8452"}
    for label in (-1, 1):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[label],
            s=28,
            alpha=0.9,
            label=class_names[label],
            edgecolors="none",
        )
    support = model["alpha"] > SUPPORT_TOL
    ax.scatter(
        X[support, 0],
        X[support, 1],
        facecolors="none",
        edgecolors="black",
        s=110,
        linewidths=1.2,
        label="опорные векторы",
    )
    colorbar = fig.colorbar(fill, ax=ax)
    colorbar.set_label("отступ f(x)")
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.extend(
        [
            Line2D([0], [0], color="black", lw=1.6, label="граница f = 0"),
            Line2D([0], [0], color="black", lw=1.0, ls="--", label="зазор f = ±1"),
        ]
    )
    legend_labels.extend(["граница f = 0", "зазор f = ±1"])
    ax.legend(legend_handles, legend_labels, loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"рисунок: {path}")


def print_baseline(y_train, y_test):
    majority = -1 if np.sum(y_train == -1) >= np.sum(y_train == 1) else 1
    accuracy = float(np.mean(y_test == majority))
    recall_forged = float(np.sum((y_test == 1) & (y_test == majority)) / np.sum(y_test == 1))
    print(
        f"база «всегда класс {majority}»: accuracy на test {accuracy:.4f}, "
        f"recall+1 {recall_forged:.4f}"
    )


def main():
    prepared = prepare_data(load_raw_dataframe())
    X_train, X_test, y_train, y_test = as_numpy(prepared)
    print("=== Пункты 2 и 4. Линейный SVM, купюры, 4 признака, C = 1 ===")
    print_baseline(y_train, y_test)
    linear = fit_model(X_train, y_train, C=C_SOFT_MARGIN, kernel_name="linear")
    print_solver_status(linear)
    print_linear_classifier(linear, prepared["feature_names"])
    print_split_metrics("train", y_train, linear, X_train)
    print_split_metrics("test", y_test, linear, X_test)
    print("=== Пункт 6. Эталон SVC, те же купюры и то же C ===")
    compare_with_svc(linear, X_train, y_train, X_test, y_test)

    print("=== Пункт 5. Линейная граница на плоскости дисперсия–асимметрия ===")
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = visual_arrays(prepared)
    linear_2d = fit_model(X_train_2d, y_train_2d, C=C_SOFT_MARGIN, kernel_name="linear")
    print_solver_status(linear_2d)
    print_split_metrics("test 2d", y_test_2d, linear_2d, X_test_2d)
    compare_with_svc(linear_2d, X_train_2d, y_train_2d, X_test_2d, y_test_2d)
    plot_solution(
        linear_2d,
        X_train_2d,
        y_train_2d,
        PLOTS_DIR / "07_banknote_linear_2d.png",
        "Линейный SVM, C = 1, обучающие купюры",
        PLOT_NAMES["variance"],
        PLOT_NAMES["skewness"],
        CLASS_NAMES,
    )

    print("=== Пункт 3. Трюк с ядром на кольцах ===")
    circles = prepare_circles()
    ring_train = circles["X_train"]
    ring_test = circles["X_test"]
    ring_y_train = circles["y_train"]
    ring_y_test = circles["y_test"]
    gamma = gamma_after_scaling(ring_train)
    print(f"gamma = {gamma:.6f}")
    ring_names = {-1: "класс −1", 1: "класс +1"}
    linear_rings = fit_model(ring_train, ring_y_train, C=C_SOFT_MARGIN, kernel_name="linear")
    print("линейное ядро на кольцах, test:")
    print_split_metrics("test", ring_y_test, linear_rings, ring_test)
    plot_solution(
        linear_rings,
        ring_train,
        ring_y_train,
        PLOTS_DIR / "08_circles_linear.png",
        "Кольца, линейное ядро, C = 1",
        "x1",
        "x2",
        ring_names,
    )
    rbf_rings = fit_model(
        ring_train,
        ring_y_train,
        C=C_SOFT_MARGIN,
        kernel_name="rbf",
        gamma=gamma,
    )
    print("RBF на кольцах:")
    print_solver_status(rbf_rings)
    print_split_metrics("train", ring_y_train, rbf_rings, ring_train)
    print_split_metrics("test", ring_y_test, rbf_rings, ring_test)
    print("=== Пункт 6. Эталон SVC, те же кольца, тот же RBF и тот же gamma ===")
    compare_with_svc(rbf_rings, ring_train, ring_y_train, ring_test, ring_y_test)
    plot_solution(
        rbf_rings,
        ring_train,
        ring_y_train,
        PLOTS_DIR / "09_circles_rbf.png",
        f"Кольца, RBF, C = 1, gamma = {gamma:.2f}",
        "x1",
        "x2",
        ring_names,
    )


if __name__ == "__main__":
    main()
