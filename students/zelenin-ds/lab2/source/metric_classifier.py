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
        self.order = None            # индексы соседей по возрастанию расстояния
        self.wrong = None            # True, если сосед чужого класса
        self.compact_profile = None  # шаг 6
        self.error = None            # шаг 7

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
        labels = y[self.order]
        self.wrong = y[:, None] != labels
        print("order:", self.order.shape, "wrong:", self.wrong.shape)

    def _profile(self):
        return self.wrong.mean(axis=0)

    def _error(self, wrong=None, k=None):
        wrong = self.wrong if wrong is None else wrong
        k = self.k if k is None else k
        bad = wrong[:, :k].sum(axis=1) > k / 2
        return bad.mean(), bad

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self._build_neighbors(X, y)
        self.compact_profile = self._profile()
        self.error, self.bad = self._error()
        print("профиль[0] (должен ≈ LOO 1-NN):", self.compact_profile[0])
        print(f"ошибка при k={self.k}:", self.error)
        return self

    def error_with_prototype(self, alive):
        errors = 0
        for i in range(len(self.y)):
            row = self.order[i]  # соседи объекта i по возрастанию расстояния
            row = row[alive[row]]  # оставляем только живых
            nb = row[:self.k]  # первые k живых соседей
            wrong_count = np.sum(self.y[nb] != self.y[i])
            if wrong_count > self.k / 2:  # большинство чужое -> ошибка
                errors += 1
        return errors / len(self.y)

    def select(self, max_steps=None):
        n = len(self.y)
        alive = np.ones(n, dtype=bool)
        history = []
        best_err = self.error_with_prototype(alive)

        for step in range(max_steps or n - 2):
            best_i, best_new = None, None
            for i in np.where(alive)[0]:
                alive[i] = False
                err = self.error_with_prototype(alive)
                alive[i] = True
                if best_new is None or err < best_new:
                    best_i, best_new = i, err

            if best_new > best_err + 0.001:
                break
            alive[best_i] = False
            best_err = best_new
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


