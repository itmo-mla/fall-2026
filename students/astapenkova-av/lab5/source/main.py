import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42
TEST_SIZE = 0.25
C = 1.0
MAX_ITER = 100
TOL = 1e-12
IMAGES_DIR = Path(__file__).resolve().parent / "images"


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def train_test_split_numpy(X, y, test_size, random_state):
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(X.shape[0])
    n_test = int(round(X.shape[0] * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def accuracy_score_numpy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def log_loss_numpy(y_true, proba):
    eps = 1e-15
    proba = np.clip(proba, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)))


def confusion_matrix_numpy(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


class BaseLogisticRegression:
    def __init__(self, C=1.0, max_iter=100, tol=1e-10):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.loss_history_ = []
        self.weights_history_ = []
        self.step_history_ = []
        self.n_iter_ = 0

    def penalty_matrix(self, n_params):
        penalty = np.eye(n_params) / self.C
        penalty[0, 0] = 0.0
        return penalty

    def objective(self, Xb, y, w):
        z = Xb @ w
        data_term = np.sum(np.logaddexp(0.0, z) - y * z)
        penalty_term = 0.5 / self.C * np.sum(w[1:] ** 2)
        return float(data_term + penalty_term)

    def step(self, Xb, y, w):
        raise NotImplementedError

    def fit(self, X, y):
        Xb = add_intercept(X)
        w = np.zeros(Xb.shape[1])
        self.loss_history_ = [self.objective(Xb, y, w)]
        self.weights_history_ = [w.copy()]
        self.step_history_ = []
        self.n_iter_ = 0
        for iteration in range(self.max_iter):
            w_new = self.step(Xb, y, w)
            step_norm = float(np.linalg.norm(w_new - w))
            w = w_new
            self.n_iter_ = iteration + 1
            self.loss_history_.append(self.objective(Xb, y, w))
            self.weights_history_.append(w.copy())
            self.step_history_.append(step_norm)
            if step_norm < self.tol:
                break
        self.weights_ = w
        return self

    def decision_function(self, X):
        return add_intercept(X) @ self.weights_

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class NewtonRaphsonLogisticRegression(BaseLogisticRegression):
    def step(self, Xb, y, w):
        p = sigmoid(Xb @ w)
        penalty = self.penalty_matrix(Xb.shape[1])
        gradient = Xb.T @ (p - y) + penalty @ w
        s = p * (1.0 - p)
        hessian = Xb.T @ (Xb * s[:, None]) + penalty
        delta = np.linalg.solve(hessian, gradient)
        return w - delta


class IRLSLogisticRegression(BaseLogisticRegression):
    def step(self, Xb, y, w):
        eta = Xb @ w
        p = sigmoid(eta)
        s = np.clip(p * (1.0 - p), 1e-12, None)
        z = eta + (y - p) / s
        penalty = self.penalty_matrix(Xb.shape[1])
        A = Xb.T @ (Xb * s[:, None]) + penalty
        b = Xb.T @ (s * z)
        return np.linalg.solve(A, b)


def fit_reference(X, y, C):
    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        tol=1e-12,
        max_iter=100000,
    )
    model.fit(X, y)
    weights = np.concatenate([model.intercept_, model.coef_.ravel()])
    return model, weights


def print_metrics(name, y_true, y_pred, proba):
    print(f"{name}")
    print(f"  accuracy: {accuracy_score_numpy(y_true, y_pred):.6f}")
    print(f"  log loss: {log_loss_numpy(y_true, proba):.6f}")
    print(f"  confusion matrix:\n{confusion_matrix_numpy(y_true, y_pred)}")


def plot_convergence(newton, irls, reference_loss):
    eps = 1e-16
    newton_gap = np.abs(np.array(newton.loss_history_) - reference_loss) + eps
    irls_gap = np.abs(np.array(irls.loss_history_) - reference_loss) + eps
    plt.figure(figsize=(9, 6))
    plt.plot(range(len(newton_gap)), newton_gap, marker="o", label="Newton-Raphson")
    plt.plot(range(len(irls_gap)), irls_gap, marker="s", linestyle="--", label="IRLS")
    plt.yscale("log")
    plt.xlabel("Номер итерации")
    plt.ylabel("|J(w_k) - J(w_sklearn)|")
    plt.title("Сходимость функционала по итерациям")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "convergence.png", dpi=150)
    plt.close()


def plot_step_norms(newton, irls):
    eps = 1e-16
    newton_steps = np.array(newton.step_history_) + eps
    irls_steps = np.array(irls.step_history_) + eps
    plt.figure(figsize=(9, 6))
    plt.plot(range(1, len(newton_steps) + 1), newton_steps, marker="o", label="Newton-Raphson")
    plt.plot(range(1, len(irls_steps) + 1), irls_steps, marker="s", linestyle="--", label="IRLS")
    plt.yscale("log")
    plt.xlabel("Номер итерации")
    plt.ylabel("||w_k - w_(k-1)||")
    plt.title("Норма шага по итерациям")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "step_norms.png", dpi=150)
    plt.close()


def plot_weights(w_newton, w_irls, w_reference):
    indices = np.arange(len(w_reference))
    width = 0.27
    plt.figure(figsize=(12, 6))
    plt.bar(indices - width, w_newton, width=width, label="Newton-Raphson")
    plt.bar(indices, w_irls, width=width, label="IRLS")
    plt.bar(indices + width, w_reference, width=width, label="sklearn")
    plt.xlabel("Индекс параметра (0 - свободный член)")
    plt.ylabel("Значение коэффициента")
    plt.title("Сравнение найденных коэффициентов")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "weights.png", dpi=150)
    plt.close()


def plot_probabilities(proba_newton, proba_irls, proba_reference):
    plt.figure(figsize=(7, 7))
    plt.scatter(proba_reference, proba_newton, s=25, alpha=0.7, label="Newton-Raphson")
    plt.scatter(proba_reference, proba_irls, s=25, alpha=0.7, marker="x", label="IRLS")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="y = x")
    plt.xlabel("Вероятность класса 1 (sklearn)")
    plt.ylabel("Вероятность класса 1 (собственная реализация)")
    plt.title("Сравнение вероятностей на тестовой выборке")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "probabilities.png", dpi=150)
    plt.close()


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()
    X = data.data
    y = data.target.astype(float)
    print(f"Объектов: {X.shape[0]}, признаков: {X.shape[1]}")
    print(f"Доля класса 1: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X, y, TEST_SIZE, RANDOM_STATE
    )
    X_train, X_test = standardize(X_train, X_test)
    print(f"Обучающая выборка: {X_train.shape[0]}, тестовая выборка: {X_test.shape[0]}")

    newton = NewtonRaphsonLogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL)
    newton.fit(X_train, y_train)

    irls = IRLSLogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL)
    irls.fit(X_train, y_train)

    reference, w_reference = fit_reference(X_train, y_train, C)

    print(f"\nИтераций Newton-Raphson: {newton.n_iter_}")
    print(f"Итераций IRLS: {irls.n_iter_}")

    Xb_train = add_intercept(X_train)
    reference_loss = newton.objective(Xb_train, y_train, w_reference)
    print(f"\nФункционал (Newton-Raphson): {newton.loss_history_[-1]:.12f}")
    print(f"Функционал (IRLS): {irls.loss_history_[-1]:.12f}")
    print(f"Функционал (sklearn): {reference_loss:.12f}")

    proba_newton = newton.predict_proba(X_test)
    proba_irls = irls.predict_proba(X_test)
    proba_reference = reference.predict_proba(X_test)[:, 1]

    pred_newton = newton.predict(X_test)
    pred_irls = irls.predict(X_test)
    pred_reference = reference.predict(X_test)

    print("\nСравнение коэффициентов")
    print(f"  max|w_newton - w_sklearn|: {np.max(np.abs(newton.weights_ - w_reference)):.3e}")
    print(f"  max|w_irls - w_sklearn|: {np.max(np.abs(irls.weights_ - w_reference)):.3e}")
    print(f"  max|w_newton - w_irls|: {np.max(np.abs(newton.weights_ - irls.weights_)):.3e}")

    print("\nСравнение вероятностей на тесте")
    print(f"  max|p_newton - p_sklearn|: {np.max(np.abs(proba_newton - proba_reference)):.3e}")
    print(f"  max|p_irls - p_sklearn|: {np.max(np.abs(proba_irls - proba_reference)):.3e}")
    print(f"  max|p_newton - p_irls|: {np.max(np.abs(proba_newton - proba_irls)):.3e}")

    print("\nСовпадение предсказанных классов на тесте")
    print(f"  Newton-Raphson и sklearn: {np.mean(pred_newton == pred_reference):.6f}")
    print(f"  IRLS и sklearn: {np.mean(pred_irls == pred_reference):.6f}")
    print(f"  Newton-Raphson и IRLS: {np.mean(pred_newton == pred_irls):.6f}")

    print("\nМетрики на тестовой выборке")
    print_metrics("Newton-Raphson", y_test, pred_newton, proba_newton)
    print_metrics("IRLS", y_test, pred_irls, proba_irls)
    print_metrics("sklearn", y_test, pred_reference, proba_reference)

    plot_convergence(newton, irls, reference_loss)
    plot_step_norms(newton, irls)
    plot_weights(newton.weights_, irls.weights_, w_reference)
    plot_probabilities(proba_newton, proba_irls, proba_reference)
    print(f"\nГрафики сохранены в {IMAGES_DIR}")


if __name__ == "__main__":
    main()