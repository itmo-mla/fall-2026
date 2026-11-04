import numpy as np

from kernels import Kernel, GaussianKernel


class ParzenClassifier:

    def __init__(self):
        self.y = None
        self.X: np.ndarray | None = None

    def calculate_metrics(self, x, b, p=2) -> np.ndarray:
        return np.pow(np.sum(np.abs(x[:, None, :] - b[None, :, :]) ** p,
                             axis=2), 1 / p)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError("Количество объектов X должно совпадать с количеством меток y")
        self.X = X
        self.y = y

    def predict(self, x: np.ndarray, kernel: Kernel, k, p=2) -> np.ndarray:
        dists = self.calculate_metrics(x, self.X, p)
        sorted_dists = np.sort(dists, axis=1)
        h = sorted_dists[:, k - 1]

        r = dists / h[:, None]
        scores = []
        y_un = np.unique(self.y)
        for y_i in y_un:
            mask = self.y == y_i
            class_scores = np.sum(kernel(r[:, mask]), axis=1)
            scores.append(class_scores)

        pred = np.argmax(scores, axis=0)
        return y_un[pred]


class PrototypeSelection:
    def __init__(self, k: int = 7):
        if k % 2 == 0:
            raise ValueError("нечётное k, чтобы не было ничьих")
        self.k = k
        self.X = None
        self.y = None
        self.order = None
        self.wrong = None
        self.compact_profile = None
        self.error = None

    def calculate_metrics(self, x, b, p=2) -> np.ndarray:
        if p == 2:
            sq_x = (x ** 2).sum(1)[:, None]
            sq_b = (b ** 2).sum(1)[None, :]
            return np.sqrt(np.maximum(sq_x + sq_b - 2 * x @ b.T, 0))
        return np.power(
            np.sum(np.abs(x[:, None, :] - b[None, :, :]) ** p, axis=2), 1 / p)

    def _build_neighbors(self, X, y):
        dists = self.calculate_metrics(X, X)
        np.fill_diagonal(dists, np.inf)
        self.order = dists.argsort(axis=1)[:, :-1]
        self.d_sorted = np.sort(dists, axis=1)[:, :-1]
        labels = y[self.order]
        self.wrong = y[:, None] != labels
        print("order:", self.order.shape, "wrong:", self.wrong.shape)

    def _profile(self):
        return self.wrong.mean(axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self._build_neighbors(X, y)
        self.compact_profile = self._profile()
        self.error = self.error_with_prototype(np.ones(len(y), dtype=bool))
        print("профиль[0] (должен ≈ LOO 1-NN):", self.compact_profile[0])
        print(f"CCV-критерий при k={self.k}:", self.error)
        return self

    def error_with_prototype(self, alive):
        return self._contributions(alive).mean()

    def _contributions(self, alive, rows=None):
        order = self.order if rows is None else self.order[rows]
        wrong = self.wrong if rows is None else self.wrong[rows]
        d_all = self.d_sorted if rows is None else self.d_sorted[rows]
        A = alive[order]
        rank = np.cumsum(A, axis=1)
        use = A & (rank <= self.k)
        d = np.where(use, d_all, 0.0)
        h = np.maximum(d.max(axis=1), 1e-12)
        w = np.where(use, np.exp(-2 * (d / h[:, None]) ** 2), 0.0)
        pi = w / np.maximum(w.sum(axis=1), 1e-12)[:, None]
        return (pi * (use & wrong)).sum(axis=1)

    def select(self, max_steps=None):
        n = len(self.y)
        alive = np.ones(n, dtype=bool)
        history = []
        c = self._contributions(alive)
        base_sum = c.sum()
        best_err = base_sum / n

        for step in range(max_steps or n - 2):
            A = alive[self.order]
            rank = np.cumsum(A, axis=1)
            use = A & (rank <= self.k)
            affected = [np.zeros(0, dtype=int) for _ in range(n)]
            for p in range(min(n - 1, self.k + step)):
                rows = np.nonzero(use[:, p])[0]
                cols = self.order[rows, p]
                for i, u in zip(cols, rows):
                    old = affected[i]
                    affected[i] = np.concatenate([old, [u]]) if len(old) else np.array([u])

            best_i, best_new = None, None
            for i in np.where(alive)[0]:
                rows = affected[i]
                if len(rows) == 0:
                    continue
                alive[i] = False
                c_new = self._contributions(alive, rows)
                alive[i] = True
                err = (base_sum - c[rows].sum() + c_new.sum()) / n
                if best_new is None or err < best_new:
                    best_i, best_new = i, err

            if best_new is None or best_new > best_err:
                break
            alive[best_i] = False
            c = self._contributions(alive)
            base_sum = c.sum()
            best_err = base_sum / n
            history.append((alive.sum(), best_err))
            print(step, alive.sum(), best_err)

        self.alive = alive
        self.history = history
        return alive

    def predict(self, X_new):
        P = self.X[self.alive]
        yP = self.y[self.alive]
        d = self.calculate_metrics(X_new, P)
        nb = d.argsort(axis=1)[:, :self.k]
        votes = (yP[nb] == 1).sum(axis=1)
        return np.where(votes > self.k / 2, 1, -1)


