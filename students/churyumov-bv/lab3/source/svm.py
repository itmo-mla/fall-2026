import numpy as np
from scipy.optimize import minimize


# Ядра (слайды 14-16). Каждое ядро принимает две выборки и возвращает матрицу Грама K[i, j] = K(x_i, z_j).
def linear_kernel():
    return lambda X, Z: X @ Z.T                                 # K(x, x') = <x, x'>


def poly_kernel(d):
    return lambda X, Z: (X @ Z.T + 1) ** d                      # K(x, x') = (<x, x'> + 1)^d


def rbf_kernel(gamma):
    def K(X, Z):
        sq = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * X @ Z.T
        return np.exp(-gamma * np.maximum(sq, 0))               # K(x, x') = exp(-gamma ||x - x'||^2)
    return K


def solve_dual(Kmat, y, C):
    r"""
    Двойственная задача (слайд 12):
        -L(lambda) = -sum_i lambda_i + 1/2 sum_i sum_j lambda_i lambda_j y_i y_j K(x_i, x_j) -> min_lambda
        sum_i lambda_i y_i = 0,   0 <= lambda_i <= C

    Kmat - матрица Грама K(x_i, x_j), shape (l, l)
    y    - метки из {-1, +1}, shape (l,)
    C    - параметр регуляризации
    """
    l = len(y)
    Q = (y[:, None] * y[None, :]) * Kmat                       # Q_ij = y_i y_j K(x_i, x_j)

    res = minimize(
        fun=lambda lam: 0.5 * lam @ Q @ lam - lam.sum(),       # -L(lambda)
        x0=np.zeros(l),
        jac=lambda lam: Q @ lam - 1,                           # градиент: Q lambda - 1
        bounds=[(0, C)] * l,                                   # 0 <= lambda_i <= C
        constraints=[{"type": "eq",                            # sum_i lambda_i y_i = 0
                      "fun": lambda lam: lam @ y,
                      "jac": lambda lam: y}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return res.x


class SVM:
    """SVM: решение двойственной задачи по lambda, нелинейность - за счёт ядра K."""

    def __init__(self, kernel, C=1.0, eps=1e-6):
        self.kernel = kernel
        self.C = C
        self.eps = eps                                         # lambda_i > eps считаем ненулевой

    def fit(self, X, y):
        """y - метки из {-1, +1}"""
        lam = solve_dual(self.kernel(X, X), y, self.C)
        lam[lam < self.eps] = 0                                # опорные объекты: lambda_i != 0 (слайд 10)
        lam[lam > self.C - self.eps] = self.C

        self.X, self.y, self.lam = X, y, lam
        self.sv = lam > 0                                      # опорные объекты
        self.boundary = self.sv & (lam < self.C)               # опорные-граничные: 0 < lambda_i < C, M_i = 1

        # w_0 = sum_j lambda_j y_j K(x_j, x_i) - y_i для любого граничного i (слайд 12); усредняем для устойчивости
        f = self.kernel(X, X) @ (lam * y)
        idx = self.boundary if self.boundary.any() else self.sv
        self.w0 = float(np.mean(f[idx] - y[idx]))
        return self

    def decision_function(self, X):
        # sum_i lambda_i y_i K(x, x_i) - w_0; суммирование только по опорным объектам
        s = self.sv
        return self.kernel(X, self.X[s]) @ (self.lam[s] * self.y[s]) - self.w0

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)  # a(x) = sign(...)

    def margins(self):
        """M_i = y_i (sum_j lambda_j y_j K(x_i, x_j) - w_0) на обучающей выборке"""
        return self.y * self.decision_function(self.X)

    def types(self):
        """Типизация объектов (слайд 10): 0 - периферийный, 1 - опорный-граничный, 2 - опорный-нарушитель"""
        t = np.zeros(len(self.y), dtype=int)
        t[self.boundary] = 1
        t[self.sv & ~self.boundary] = 2
        return t

    def weights(self):
        """w = sum_i lambda_i y_i x_i; имеет смысл только для линейного ядра"""
        return (self.lam * self.y) @ self.X
