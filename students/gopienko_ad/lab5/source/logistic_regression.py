import numpy as np
from scipy.special import expit


class LogisticRegressionBase:
    def __init__(
            self,
            alpha: float = 0.0,
            tol: float = 1e-9,
            max_iter: int = 100,
            min_step: float = 1e-8
    ):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.min_step = min_step
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] = []
        self.grad_norm_history_: list[float] = []
        self.step_history_: list[float] = []
        self.n_iter_: int = 0
        self.converged_: bool = False

    @staticmethod
    def _matrix(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.column_stack((np.ones(x.shape[0]), x))

    def _penalty(self, beta: np.ndarray) -> np.ndarray:
        d = np.full(beta.size, self.alpha)
        d[0] = 0.0
        return d

    def _loss(self, F: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        eta = F @ beta
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * self.alpha * np.dot(beta[1:], beta[1:]))

    def _gradient(self, F: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return F.T @ (expit(F @ beta) - y) / len(y) + self._penalty(beta) * beta

    @staticmethod
    def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _direction(self, F: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        F = self._matrix(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        beta = np.zeros(F.shape[1])  # deterministic common initialization
        self.loss_history_ = [self._loss(F, y, beta)]
        self.grad_norm_history_ = [float(np.linalg.norm(self._gradient(F, y, beta)))]
        self.step_history_ = [0.0]
        self.n_iter_ = 0
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            grad = self._gradient(F, y, beta)
            if np.linalg.norm(grad, ord=np.inf) < self.tol:
                self.converged_ = True
                break
            direction = self._direction(F, y, beta)
            old_loss = self.loss_history_[-1]
            step = 1.0
            while step >= self.min_step:
                candidate = beta + step * direction
                new_loss = self._loss(F, y, candidate)
                if np.isfinite(new_loss) and new_loss <= old_loss + 1e-14:
                    break
                step *= 0.5
            else:
                if np.linalg.norm(grad, ord=np.inf) < 10 * self.tol:
                    self.converged_ = True
                break
            beta = candidate
            self.n_iter_ = iteration
            self.loss_history_.append(new_loss)
            self.grad_norm_history_.append(float(np.linalg.norm(self._gradient(F, y, beta))))
            self.step_history_.append(step)

            if np.linalg.norm(self._gradient(F, y, beta), ord=np.inf) < self.tol:
                self.converged_ = True
                break

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].copy()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Модель не обучена")
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = expit(self.decision_function(X))
        return np.column_stack((1.0 - p, p))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class LogisticRegressionNewton(LogisticRegressionBase):

    def _direction(self, F: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        p = expit(F @ beta)
        weights = p * (1.0 - p)
        H = F.T @ (weights[:, None] * F) / len(y)
        H[np.diag_indices_from(H)] += self._penalty(beta)
        gradient = self._gradient(F, y, beta)
        return -self._solve(H, gradient)

class LogisticRegressionIRLS(LogisticRegressionBase):

    def _direction(self, F: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        eta = F @ beta
        p = expit(eta)
        weights = np.maximum(p * (1.0 - p), 1e-12)
        z = eta + (y - p) / weights  # working response
        sqrt_w = np.sqrt(weights)
        FW = F * sqrt_w[:, None]
        zw = z * sqrt_w
        A = (FW.T @ FW) / len(y)
        A[np.diag_indices_from(A)] += self._penalty(beta)
        rhs = FW.T @ zw / len(y)
        beta_new = self._solve(A, rhs)
        return beta_new - beta
    