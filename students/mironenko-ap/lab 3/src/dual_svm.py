import numpy as np

from src.kernels import get_kernel


class DualSVM:
    """
        SVM через двойственную задачу
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale",
                 degree=3, coef0=1.0,
                 solver="smo",
                 tol=1e-5, tol_sv=1e-6,
                 max_passes=100, max_iter=5000,
                 n_polish_rounds=3, polish_max_passes=30,
                 polish_max_iter=1000,
                 verbose=False):
        if solver not in ("smo", "scipy"):
            raise ValueError(f"solver должен быть 'smo' или 'scipy', "
                             f"получено: {solver!r}")

        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.solver = solver
        self.tol = tol
        self.tol_sv = tol_sv
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.n_polish_rounds = n_polish_rounds
        self.polish_max_passes = polish_max_passes
        self.polish_max_iter = polish_max_iter
        self.verbose = verbose

        self.X_train = None
        self.y_train = None
        self.lam = None
        self.b = None
        self.sv_mask = None
        self._gamma_effective = None
        self.n_iter_ = None
        self.polish_iters_ = []
        self.converged_ = False
        self.max_kkt_violation_ = None

    #  ядро
    def _kernel(self, X, Y):
        if self.kernel_name == "rbf":
            return get_kernel("rbf")(X, Y, gamma=self._gamma_effective)
        if self.kernel_name == "poly":
            return get_kernel("poly")(X, Y,
                                      degree=self.degree,
                                      coef0=self.coef0)
        return get_kernel("linear")(X, Y)

    #  корректный пересчёт b по KKT
    def _compute_b(self, lam,
                   y,
                   K,
                   b_fallback=0.0):
        # Свободные SV дают точное значение b
        free = (lam > self.tol_sv) & (lam < self.C - self.tol_sv)
        if free.sum() > 0:
            K_free = K[free, :]
            return float(np.mean(y[free] - K_free @ (lam * y)))

        # Нет свободных SV — используем KKT-границы
        g = K @ (lam * y)  # f(x_i) без b

        b_low = -np.inf
        b_high = np.inf

        # λ = 0, y = 1: g_i + b ≥ 1  →  b ≥ 1 - g_i
        mask = (y == 1) & (lam <= self.tol_sv)
        if mask.any():
            b_low = max(b_low, float((1.0 - g[mask]).max()))

        # λ = C, y = -1: -(g_i + b) ≤ 1  →  b ≥ -1 - g_i
        mask = (y == -1) & (lam >= self.C - self.tol_sv)
        if mask.any():
            b_low = max(b_low, float((-1.0 - g[mask]).max()))

        # λ = 0, y = -1: -(g_i + b) ≥ 1  →  b ≤ -1 - g_i
        mask = (y == -1) & (lam <= self.tol_sv)
        if mask.any():
            b_high = min(b_high, float((-1.0 - g[mask]).min()))

        # λ = C, y = 1: g_i + b ≤ 1  →  b ≤ 1 - g_i
        mask = (y == 1) & (lam >= self.C - self.tol_sv)
        if mask.any():
            b_high = min(b_high, float((1.0 - g[mask]).min()))

        if np.isfinite(b_low) and np.isfinite(b_high) and b_low <= b_high:
            return float((b_low + b_high) / 2.0)

        # Интервал пуст или не определён — используем b из решателя
        return b_fallback

    #  кандидаты j в порядке убывания |E_i - E_j|
    @staticmethod
    def _candidate_js(i, E):
        diff = np.abs(E - E[i])
        diff[i] = -np.inf
        return np.argsort(-diff)

    #  основной SMO
    def _smo(self, y, K):
        n = len(y)
        C = self.C
        tol = self.tol

        lam = np.zeros(n)
        b = 0.0
        passes = 0
        iters = 0

        while passes < self.max_passes and iters < self.max_iter:
            num_changed = 0
            E = K @ (lam * y) + b - y

            for i in range(n):
                if not ((y[i] * E[i] < -tol and lam[i] < C) or
                        (y[i] * E[i] > tol and lam[i] > 0)):
                    continue

                made_update = False

                for j in self._candidate_js(i, E):
                    j = int(j)

                    if y[i] != y[j]:
                        L = max(0.0, lam[j] - lam[i])
                        H = min(C, C + lam[j] - lam[i])
                    else:
                        L = max(0.0, lam[i] + lam[j] - C)
                        H = min(C, lam[i] + lam[j])

                    if L >= H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    # Отрицательное значение с запасом — защита от
                    # вырожденных (линейных) ядер и деления на ноль.
                    if eta >= -1e-12:
                        continue

                    lam_i_old, lam_j_old = lam[i], lam[j]
                    b_old = b

                    lam_j_new = lam_j_old - y[j] * (E[i] - E[j]) / eta
                    lam_j_new = min(H, max(L, lam_j_new))

                    if abs(lam_j_new - lam_j_old) < 1e-9:
                        continue

                    lam_i_new = lam_i_old + y[i] * y[j] * (
                            lam_j_old - lam_j_new
                    )

                    b1 = b - E[i] \
                         - y[i] * (lam_i_new - lam_i_old) * K[i, i] \
                         - y[j] * (lam_j_new - lam_j_old) * K[i, j]
                    b2 = b - E[j] \
                         - y[i] * (lam_i_new - lam_i_old) * K[i, j] \
                         - y[j] * (lam_j_new - lam_j_old) * K[j, j]

                    if 0.0 < lam_i_new < C:
                        b = b1
                    elif 0.0 < lam_j_new < C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    d_i = lam_i_new - lam_i_old
                    d_j = lam_j_new - lam_j_old
                    lam[i] = lam_i_new
                    lam[j] = lam_j_new

                    E += d_i * y[i] * K[i] + d_j * y[j] * K[j] + (b - b_old)

                    made_update = True
                    break

                if made_update:
                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0
            iters += 1

            if self.verbose:
                print(f"[SMO]   iter={iters:4d} passes={passes:3d} "
                      f"changed={num_changed:4d} "
                      f"nSV={(lam > self.tol_sv).sum():4d}")

        self.n_iter_ = iters
        return lam, b

    #  решение через scipy.optimize.minimize
    def _solve_scipy(self, y, K):
        from scipy.optimize import minimize

        n = len(y)
        C = self.C
        H = (y[:, None] * y[None, :]) * K
        # Микро-регуляризация для устойчивости SLSQP на вырожденных
        # (например, линейных) ядрах.
        H += 1e-8 * np.eye(n)

        def objective(lam):
            return 0.5 * lam @ H @ lam - np.sum(lam)

        def jac(lam):
            return H @ lam - np.ones(n)

        constraints = [{
            "type": "eq",
            "fun": lambda lam: lam @ y,
            "jac": lambda lam: y.astype(float),
        }]
        bounds = [(0.0, C) for _ in range(n)]

        res = minimize(
            objective,
            np.zeros(n),
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 5000, "ftol": 1e-12},
        )

        if self.verbose:
            print(f"[scipy] success={res.success}, "
                  f"nit={getattr(res, 'nit', 'n/a')}, msg={res.message}")

        lam = np.clip(res.x, 0.0, C)
        b = 0.0
        free = (lam > self.tol_sv) & (lam < C - self.tol_sv)
        if free.sum() > 0:
            K_free = K[free, :]
            b = float(np.mean(y[free] - K_free @ (lam * y)))

        self.n_iter_ = int(getattr(res, "nit", 0))
        return lam, b

    #  SMO с фиксированным b (polish)
    def _smo_lam_only(self, y, K, lam_init, b_fixed):
        n = len(y)
        C = self.C
        tol = self.tol

        lam = lam_init.copy()
        passes = 0
        iters = 0

        while passes < self.polish_max_passes and iters < self.polish_max_iter:
            num_changed = 0
            E = K @ (lam * y) + b_fixed - y

            for i in range(n):
                if not ((y[i] * E[i] < -tol and lam[i] < C) or
                        (y[i] * E[i] > tol and lam[i] > 0)):
                    continue

                made_update = False

                for j in self._candidate_js(i, E):
                    j = int(j)

                    if y[i] != y[j]:
                        L = max(0.0, lam[j] - lam[i])
                        H = min(C, C + lam[j] - lam[i])
                    else:
                        L = max(0.0, lam[i] + lam[j] - C)
                        H = min(C, lam[i] + lam[j])

                    if L >= H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= -1e-12:
                        continue

                    lam_i_old, lam_j_old = lam[i], lam[j]

                    lam_j_new = lam_j_old - y[j] * (E[i] - E[j]) / eta
                    lam_j_new = min(H, max(L, lam_j_new))

                    if abs(lam_j_new - lam_j_old) < 1e-9:
                        continue

                    lam_i_new = lam_i_old + y[i] * y[j] * (
                            lam_j_old - lam_j_new
                    )

                    d_i = lam_i_new - lam_i_old
                    d_j = lam_j_new - lam_j_old
                    lam[i] = lam_i_new
                    lam[j] = lam_j_new

                    E += d_i * y[i] * K[i] + d_j * y[j] * K[j]

                    made_update = True
                    break

                if made_update:
                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0
            iters += 1

            if self.verbose:
                print(f"[polish] iter={iters:4d} passes={passes:3d} "
                      f"changed={num_changed:4d} "
                      f"nSV={(lam > self.tol_sv).sum():4d}")

        return lam, iters, passes >= self.polish_max_passes

    #  обучение
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        if self.gamma == "scale":
            self._gamma_effective = 1.0 / (X.shape[1] * X.var())
        else:
            self._gamma_effective = float(self.gamma)

        if self.verbose and self.kernel_name == "rbf":
            print(f"[{self.solver}] gamma_effective = "
                  f"{self._gamma_effective:.6f}")

        K = self._kernel(X, X)

        #  Phase 1: основной решатель
        if self.solver == "smo":
            lam, b_solver = self._smo(y, K)
        else:
            lam, b_solver = self._solve_scipy(y, K)

        #  Phase 2: polish
        self.polish_iters_ = []
        for polish_round in range(self.n_polish_rounds):
            free = (lam > self.tol_sv) & (lam < self.C - self.tol_sv)
            if free.sum() == 0:
                break
            K_free = K[free, :]
            b_new = float(np.mean(y[free] - K_free @ (lam * y)))
            if abs(b_new - b_solver) < self.tol:
                b_solver = b_new
                break
            b_solver = b_new
            lam, iters, conv = self._smo_lam_only(y, K, lam, b_solver)
            self.polish_iters_.append(iters)

        #  Выбор финального b: два кандидата + KKT-контроль
        self.lam = lam
        self.sv_mask = self.lam > self.tol_sv

        b_computed = self._compute_b(lam, y, K, b_fallback=b_solver)

        def kkt_of(b):
            E = K @ (lam * y) + b - y
            yE = y * E
            is_zero = lam <= self.tol_sv
            is_C = lam >= self.C - self.tol_sv
            is_free = ~is_zero & ~is_C

            viol = np.zeros_like(lam)
            viol[is_free] = np.abs(yE[is_free])
            viol[is_zero] = np.maximum(0.0, -yE[is_zero])
            viol[is_C] = np.maximum(0.0, yE[is_C])
            return float(viol.max())

        v_computed = kkt_of(b_computed)
        v_solver = kkt_of(b_solver)

        if v_computed <= v_solver:
            self.b = b_computed
            self.max_kkt_violation_ = v_computed
        else:
            self.b = b_solver
            self.max_kkt_violation_ = v_solver

        self.converged_ = self.max_kkt_violation_ <= max(self.tol, 1e-4)

        if self.verbose:
            print(f"[{self.solver}] max KKT violation = "
                  f"{self.max_kkt_violation_:.3e} "
                  f"(tol={self.tol:.1e}, "
                  f"converged={self.converged_})")

        return self

    #  предсказание
    def decision_function(self, X):
        Kq = self._kernel(X, self.X_train)
        return Kq @ (self.lam * self.y_train) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def n_support_vectors(self):
        return int(self.sv_mask.sum())