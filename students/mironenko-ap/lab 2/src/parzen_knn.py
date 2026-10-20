"""KNN с ядерным окном Парзена переменной ширины (sample-point)."""
import numpy as np


class ParzenKNN:
    """
    Многоклассовый KNN с гауссовым ядром Парзена и переменной шириной окна.

    Для каждого ОБУЧАЮЩЕГО объекта x_i задаётся своя ширина:
        h_i = расстояние от x_i до его k-го ближайшего соседа (сам x_i исключён).

    Оценка плотности класса c:
        p(x|c) = (1/N_c) Σ_{i: y_i=c} (1/h_i^d) K( (x - x_i) / h_i )

    где K(u) = exp(-||u||²/2) — гауссово ядро (без нормировочной константы,
    она одинакова для всех классов).

    Предсказание: argmax_c p(x|c). Априорные вероятности не нужны:
    они сокращаются с 1/N_c (π_c = N_c/N).
    """

    def __init__(self, k: int = 5, kernel: str = "gaussian"):
        assert kernel == "gaussian", kernel
        self.k = int(k)
        self.kernel = kernel

    # -------------------- обучение --------------------
    def fit(self, X, y, h=None):
        """
        h: необязательный предвычисленный массив ширин той же длины, что X.
           Полезно в LOO, чтобы не пересчитывать h на каждой свёртке.
        """
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        self.d = self.X.shape[1]
        self.h_ = self._compute_bandwidth(self.X) if h is None \
                  else np.asarray(h, dtype=float)
        return self

    def _compute_bandwidth(self, X):
        """h_i = расстояние от x_i до k-го ближайшего соседа (self исключён)."""
        n = len(X)
        diffs = X[:, None, :] - X[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diffs, diffs)
        np.fill_diagonal(sq, np.inf)          # self не считаем соседом
        kth = min(self.k, n - 1)
        kth_vals = np.partition(sq, kth - 1, axis=1)[:, kth - 1]
        return np.sqrt(np.maximum(kth_vals, 1e-24))

    # -------------------- предсказание --------------------
    def _log_weights(self, x):
        """
        log[ (1/h_i^d) * exp(-||x - x_i||² / (2 h_i²)) ] для всех обучающих i.
        Работаем в логах — иначе при d=11 величины 1/h_i^d выходят за пределы
        представимости float64.
        """
        diffs = self.X - x
        sq = np.einsum("ij,ij->i", diffs, diffs)
        return -self.d * np.log(self.h_) - 0.5 * sq / (self.h_ ** 2)

    @staticmethod
    def _logsumexp(a):
        m = a.max()
        return m + np.log(np.exp(a - m).sum())

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        preds = np.empty(len(X), dtype=self.y.dtype)
        n_classes = len(self.classes_)

        for i, x in enumerate(X):
            log_w = self._log_weights(x)
            log_scores = np.empty(n_classes)
            for j, c in enumerate(self.classes_):
                log_scores[j] = self._logsumexp(log_w[self.y == c])
            preds[i] = self.classes_[np.argmax(log_scores)]
        return preds

    def score(self, X, y):
        return (self.predict(X) == y).mean()