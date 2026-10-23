import numpy as np
from scipy.optimize import minimize, LinearConstraint


def solve_dual(K, y, C):
    n = len(y)
    yyK = np.outer(y, y) * K

    def f(lam):
        return -lam.sum() + 0.5 * lam @ yyK @ lam

    def grad(lam):
        return yyK @ lam - 1

    bounds = [(0, C)] * n
    constraint = LinearConstraint(y.reshape(1, -1), 0, 0)
    lam0 = np.zeros(n)
    return minimize(f, lam0, jac=grad, bounds=bounds, constraints=[constraint], method="SLSQP")


class DualSVM:
    def __init__(self, kernel, C=1.0, eps=1e-5, **kernel_params):
        self.kernel = kernel
        self.C = C
        self.eps = eps
        self.kernel_params = kernel_params

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X, **self.kernel_params)
        self.result = solve_dual(K, y, self.C)

        lam = self.result.x.copy()
        lam[lam < self.eps] = 0.0
        self.lam = lam

        sv = lam > 0
        self.sv_idx = np.where(sv)[0]
        self.coef = lam * y

        margin_mask = sv & (lam < self.C - self.eps)
        self.margin_idx = np.where(margin_mask)[0]
        self.violator_idx = np.where(sv & ~margin_mask)[0]

        on_margin = margin_mask if margin_mask.any() else sv
        scores = K[on_margin] @ self.coef
        self.w0 = np.mean(scores - y[on_margin])
        return self

    def decision_function(self, X):
        K = self.kernel(X, self.X_train[self.sv_idx], **self.kernel_params)
        return K @ self.coef[self.sv_idx] - self.w0

    def predict(self, X):
        return np.sign(self.decision_function(X))


def to_primal_weights(svm):
    w = svm.X_train[svm.sv_idx].T @ svm.coef[svm.sv_idx]
    return w, svm.w0


def kkt_report(svm):
    lam = svm.lam
    bounds_ok = bool(np.all(lam >= -1e-6) and np.all(lam <= svm.C + 1e-6))
    equality_gap = abs(np.sum(lam * svm.y_train))
    sv_fraction = len(svm.sv_idx) / len(svm.y_train)
    return {
        "bounds_ok": bounds_ok,
        "equality_gap": equality_gap,
        "sv_fraction": sv_fraction,
        "success": bool(svm.result.success),
        "dual_objective": float(svm.result.fun),
    }
