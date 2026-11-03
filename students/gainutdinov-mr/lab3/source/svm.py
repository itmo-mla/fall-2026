"""SVM: решение двойственной задачи с линейным или полиномиальным ядром."""

import numpy as np
from kernels import ConstantKernel, LinearKernel, PolynomialKernel, RBFKernel
from scipy import optimize


def linear_kernel(X, Z):
    """Матрица попарных скалярных произведений двух выборок."""
    return X @ Z.T


def polynomial_kernel(X, Z, degree=2):
    return (linear_kernel(X, Z) + 1) ** degree


synth_kernel = 2 * PolynomialKernel(degree=2) + ConstantKernel(3)


class SVM:
    """Бинарный SVM для меток -1 и +1.

    kernel — функция двух матриц (n, d) и (m, d), возвращающая (n, m).
    Для своего ядра нужно обеспечить положительную полуопределённость.
    Например, kernel=lambda X, Z: polynomial_kernel(X, Z, degree=3).
    Обученная модель хранит выборку и матрицу ядра для изучения решения;
    для предсказаний используются только опорные объекты.
    """

    def __init__(self, C=1.0, kernel=linear_kernel, eps=1e-5, maxiter=1000):
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C must be finite and positive")
        if not np.isfinite(eps) or not 0 < eps < C / 2:
            raise ValueError("eps must be positive and smaller than C / 2")
        if not callable(kernel):
            raise TypeError("kernel must be callable")
        self.C = C
        self.kernel = kernel
        self.eps = eps
        self.maxiter = maxiter
        self._fitted = False

    def fit(self, X, y):
        """Найти lambda, опорные объекты и свободный член и вернуть self."""
        self._fitted = False
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X = X.copy()
        self.y = y.copy()
        self.kernel_matrix = np.asarray(self.kernel(self.X, self.X), dtype=float)

        self.optimization_result = self._optimize()
        if not self.optimization_result.success:
            raise RuntimeError(
                f"Optimization failed: {self.optimization_result.message}"
            )
        self.lambdas = self.optimization_result.x
        self.support_indices = np.flatnonzero(
            self.lambdas > self.eps
        )  # может лучше попробовать хранить только опорные элементы?
        self.X_support = self.X[self.support_indices]
        self.y_support = self.y[self.support_indices]
        self.support_coefficients = self.lambdas[self.support_indices] * self.y_support
        self.w0 = self._compute_w0()
        # Явный вектор в исходном пространстве есть только для линейного ядра.
        self.weights = (
            self.X_support.T @ self.support_coefficients
            if self.kernel is linear_kernel # TODO: Только ли для linear_kernel стоит так считать, может и для каких-то других тоже разрешить?
            else None
        )
        self._fitted = True
        return self

    def _objective_and_gradient(self, lambdas):
        """
        Наша главная функция - двойственная задача по лямбда,
        выведенная из системы ККТ
        """
        alpha = lambdas * self.y
        K_alpha = self.kernel_matrix @ alpha
        objective = -lambdas.sum() + 0.5 * (alpha @ K_alpha)
        gradient = self.y * K_alpha - 1
        return objective, gradient

    def _optimize(self):
        """
        Пока что будем использовать minimize от scipy,
        Но вообще говорят можно использовать специализированный решатель задач
        квадратичного программирования
        # https://osqp.org/docs/examples/svm.html
        """
        # Q = self.y[:, None] * self.kernel_matrix * self.y[None, :] # - по сути это наш гессиан
        
        return optimize.minimize(
            self._objective_and_gradient,
            np.zeros(len(self.X)),
            jac=True,
            # L-BFGS-B, BFGS и Newton-CG - использовать нельзя
            method="SLSQP",  # вроде именно этот оптимизатор хорошо работает с ограничениями
            # trust-constr
            # 
            # method="trust-constr",
            bounds=optimize.Bounds(0, self.C),
            constraints=optimize.LinearConstraint(self.y, lb=0, ub=0),
            options={
                "maxiter": self.maxiter,
                "ftol": 1e-8, # У trust-constr нет ftol, есть gtol и verbose: 2
                # "gtol": 1e-6,
                # "verbose": 3
            },
        )

    def _compute_w0(self):
        # Сохраняем исходную конвенкцию: f(x) = sum(lambda_i y_i K(x, x_i)) - w0.
        scores = self.kernel_matrix[:, self.support_indices] @ self.support_coefficients
        candidates = scores - self.y
        at_zero = self.lambdas <= self.eps
        at_C = self.lambdas >= self.C - self.eps
        margin_mask = ~at_zero & ~at_C
        if np.any(margin_mask):
            return float(np.mean(candidates[margin_mask]))

        # Если свободных опорных объектов нет, условия ККТ задают интервал w0.
        lower = ((self.y == -1) & at_zero) | ((self.y == 1) & at_C)
        upper = ((self.y == 1) & at_zero) | ((self.y == -1) & at_C)
        return float((np.max(candidates[lower]) + np.min(candidates[upper])) / 2)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit before using the model")

    def decision_function(self, X):
        """Значения решающей функции для одного объекта или матрицы объектов."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        scores = self.kernel(X, self.X_support) @ self.support_coefficients - self.w0
        return scores[0] if single else scores

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def margin_width(self):
        """Ширина 2 / ||w|| в пространстве признаков выбранного ядра."""
        self._check_fitted()
        support_kernel = self.kernel_matrix[
            np.ix_(self.support_indices, self.support_indices)
        ]
        norm_squared = (
            self.support_coefficients @ support_kernel @ self.support_coefficients
        )
        return np.inf if norm_squared <= 0 else 2 / np.sqrt(norm_squared)


if __name__ == "__main__":
    from data import load_data, standardize_data
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    y = 2 * y - 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_train, X_test, _, _ = standardize_data(X_train, X_test)

    # Пока ради теста, не будем загружать большой датасет с авиапассажирами
    # Будем использовать - breast_cancer
    # X_train, X_test, y_train, y_test = load_data(
    #     sample_fraction=0.02, test_size=0.3, random_state=42
    # )

    model = SVM(C=1.0).fit(X_train, y_train) # Включить когда с линейным лидром все будет понятно
    # model = SVM(C=1.0).fit(X_train, y_train) # Линейное ядро - 5 минут и готово. Нелинейное - 20+ а то и часы ожидания
    print("Train accuracy:", np.mean(model.predict(X_train) == y_train))
    print("Test accuracy:", np.mean(model.predict(X_test) == y_test))
    print("Support vectors:", len(model.support_indices))
    print("Margin width:", model.margin_width())
    print("w0:", model.w0)
    print("Prediction:", model.predict(X_test[2]))
    print("True label:", y_test[2])
