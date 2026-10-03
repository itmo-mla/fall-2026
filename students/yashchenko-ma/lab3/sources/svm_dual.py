"""Шаг 2: решение двойственной задачи SVM по lambda (alpha) через scipy.optimize.minimize.

min_a  0.5 a^T Q a - 1^T a,   Q_ij = y_i y_j K(x_i, x_j)
s.t.   0 <= a_i <= C,  sum_i a_i y_i = 0

Задача квадратичная: градиент Q a - 1, гессиан Q (точные, аналитические).
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


def solve_dual(K, y, C, tol=1e-5):
    """Возвращает (alpha, результат scipy). Значения alpha < tol*C обнуляются (численный шум)."""
    n = K.shape[0]
    Q = np.outer(y, y) * K

    def objective(alpha):
        return 0.5 * alpha @ Q @ alpha - alpha.sum()

    def grad(alpha):
        return Q @ alpha - 1.0

    def hess(alpha):
        return Q

    alpha0 = np.zeros(n)
    bounds = Bounds(0.0, C)
    eq_constraint = LinearConstraint(y.reshape(1, -1), 0.0, 0.0)

    res = minimize(
        objective, alpha0, jac=grad, hess=hess,
        bounds=bounds, constraints=[eq_constraint],
        method='trust-constr',
        options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0},
    )
    # alpha = res.x
    # alpha[alpha < tol * C] = 0.0 # если  C  маленькое или значение  alpha  отрицательное из-за численной погрешности, это не гарантирует соблюдение bounds
    # so, better do
    alpha = np.asarray(res.x, dtype=float)
    alpha = np.clip(alpha, 0.0, C)
    alpha[alpha < tol * max(1.0, C)] = 0.0

    return alpha, res
