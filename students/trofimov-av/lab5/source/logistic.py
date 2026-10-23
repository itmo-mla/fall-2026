import numpy as np
from scipy.special import expit


class LogisticRegression:
    def __init__(self, method="newton", l2=1.0, max_iter=100, tol=1e-8):
        self.method = method
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError("Некорректные данные")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Нужны два класса с метками 0 и 1")
        if self.method not in ("newton", "irls"):
            raise ValueError("Неизвестный метод")
        if self.l2 < 0 or self.max_iter < 1 or self.tol <= 0:
            raise ValueError("Некорректные параметры")
        design = np.column_stack((np.ones(len(x)), x))
        penalty = np.diag(np.r_[0., np.full(x.shape[1], self.l2)])
        beta = np.zeros(design.shape[1])
        self.loss_history_ = []
        self.gradient_history_ = []
        self.converged_ = False
        self.n_iter_ = 0

        def objective(candidate):
            eta = design @ candidate
            return np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * candidate @ penalty @ candidate

        for iteration in range(self.max_iter + 1):
            eta = design @ beta
            probability = expit(eta)
            gradient = design.T @ (probability - y) + penalty @ beta
            loss = objective(beta)
            self.loss_history_.append(loss)
            self.gradient_history_.append(np.max(np.abs(gradient)))
            if self.gradient_history_[-1] <= self.tol:
                self.converged_ = True
                break
            if iteration == self.max_iter:
                break
            weights = np.maximum(probability * (1 - probability), 1e-12)
            if self.method == "newton":
                hessian = design.T @ (weights[:, None] * design) + penalty
                direction = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            else:
                working_response = eta + (y - probability) / weights
                weighted_design = np.sqrt(weights)[:, None] * design
                augmented_design = np.vstack((weighted_design, np.sqrt(penalty)))
                augmented_response = np.r_[np.sqrt(weights) * working_response, np.zeros(len(beta))]
                updated = np.linalg.lstsq(augmented_design, augmented_response, rcond=None)[0]
                direction = updated - beta
            step = 1.0
            while step > 1e-12 and objective(beta + step * direction) > loss + 1e-12:
                step *= 0.5
            if step <= 1e-12:
                raise RuntimeError("Не удалось уменьшить функцию потерь")
            beta += step * direction
            self.n_iter_ = iteration + 1

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict_proba(self, x):
        probability = expit(np.asarray(x, dtype=float) @ self.coef_ + self.intercept_)
        return np.column_stack((1 - probability, probability))

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
