"""
Собственная реализация: KNN с методом окна Парзена переменной ширины
(пункт 2 задания). Готовые реализации из библиотек не используются, только numpy.

Формула из лекции "Метрические методы классификации и регрессии" (слайд 11):

    a(x; X^l, k, K) = argmax_y sum_{i=1}^{l} [y_i = y] * K( rho(x,x_i) / rho(x, x^(k+1)) )

где x^(k+1) - (k+1)-й по счёту сосед объекта x, 
h(x) = rho(x, x^(k+1)) - переменная ширина окна (своя для каждой точки запроса). 
Ядро K - гауссово(слайд 34 лекции): K(r) = exp(-0.5 * r^2). Сумма берётся по ВСЕЙ обучающей
выборке (а не только по k ближайшим соседям) - в этом ключевое отличие
метода парзеновского окна от простого метода k ближайших соседей (слайд 8).
"""
import numpy as np


class ParzenKNN:
    """KNN с окном Парзена переменной ширины, гауссово ядро."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _pairwise_dist(self, X):
        X = np.asarray(X, dtype=float)
        return np.sqrt(np.maximum(
            ((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2), 0.0))

    def _kernel_scores(self, dists, k):
        """dists: (n_query, n_train) - матрица расстояний до всех точек
        обучающей выборки. Возвращает (n_query, n_classes) - оценки Gamma_y(x)."""
        n_query, n_train = dists.shape
        idx_sorted = np.argsort(dists, axis=1) # индексы точек в порядке возрастания расстояния: [0, 1, 2, 3, 4, 5] (расстояния тут и так идут по возрастанию)
        k_eff = min(k, n_train - 1)
        h_idx = idx_sorted[:, k_eff]                       # позиция (k+1)-го соседа ; k_eff = k = 1 - позиция k+1 в отсортированном списке (нулевая позиция — 1-й ближайший, позиция 1 — 2-й ближайший
        # h — расстояние до второго ближайшего соседа
        h = dists[np.arange(n_query), h_idx].copy()         # h(x) = ρ(x,x(k+1)), где x^(k)— k-й сосед объекта x
        h[h == 0] = 1e-9                                    # защита от деления на 0
        # вес получает каждая обучающая точка, не только ближайшие k
        weights = np.exp(-0.5 * (dists / h[:, None]) ** 2)  # веса по ВСЕЙ выборке ; (лекция 2 слайд 38): K(r), r = ρ(x,xi)/h(x), h(x) = ρ(x,x(k+1)), где x^(k)— k-й сосед объекта x

         # scores -Это вычисление Gamma_y(x)=sum_i[y_i=y] * вес_i — те самые "суммарные голоса за класс y", 
         # то есть сама формула классификации (слайд 7/11)
        scores = np.zeros((n_query, len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            scores[:, ci] = np.sum(weights * (self.y_train == c)[None, :], axis=1)
        return scores

    def predict_proba(self, X, k=None):
        k = self.k if k is None else k
        dists = self._pairwise_dist(X)
        scores = self._kernel_scores(dists, k)
        total = scores.sum(axis=1, keepdims=True)
        total[total == 0] = 1.0
        return scores / total

    def predict(self, X, k=None):
        k = self.k if k is None else k
        dists = self._pairwise_dist(X)
        scores = self._kernel_scores(dists, k)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_from_dists(self, dists, k):
        """Предсказание по заранее посчитанной матрице расстояний (для
        ускорения LOO -- см. loo.py)."""
        scores = self._kernel_scores(dists, k)
        return self.classes_[np.argmax(scores, axis=1)]


class KNNVote:
    """
    Простой метод k ближайших соседей - голосование (лекция, слайд 8):
        w(i, x) = [i <= k]
        a(x; X^l) = argmax_y sum_i [y_i = y] * w(i, x)
    В отличие от ParzenKNN, вес соседей внутри k ближайших равен 1, 
    вне - строго 0 (жёсткое окно без ядра). При k=1 это в точности классификатор
    1NN, который используется как базовый для алгоритма отбора эталонов
    (см. prototype_selection.py). Реализован без sklearn.
    """

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _pairwise_dist(self, X):
        X = np.asarray(X, dtype=float)
        return np.sqrt(np.maximum(
            ((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2), 0.0))

    def predict(self, X, k=None):
        k = self.k if k is None else k
        k = min(k, self.X_train.shape[0])
        dists = self._pairwise_dist(X)
        nn_idx = np.argsort(dists, axis=1)[:, :k]
        nn_labels = self.y_train[nn_idx]
        preds = np.empty(len(X), dtype=self.y_train.dtype)
        for i in range(len(X)):
            vals, counts = np.unique(nn_labels[i], return_counts=True)
            preds[i] = vals[np.argmax(counts)]
        return preds


if __name__ == "__main__":
    # санити-чек на simple-custom-testing данных: два хорошо разделённых кластера
    '''
    Создаётся игрушечная (simple-custom-testing) одномерная выборка: 
    три точки кластера класса 0 (около x=0) и три точки кластера класса 1 (около x=10). 
    fit просто запоминает X_train, y_train и список классов classes_ = [0, 1] (сортированный np.unique).

    Дальше вызывается predict на трёх новых точках: 
    0.05 (глубоко внутри левого кластера), 10.05 (глубоко внутри правого), 5 (ровно посередине). 
    При k=None берётся self.k=1.
    '''
    Xtr = np.array([[0], [0.1], [0.2], [10], [10.1], [10.2]])
    ytr = np.array([0, 0, 0, 1, 1, 1])
    m = ParzenKNN(k=1).fit(Xtr, ytr)
    print("предсказания:", m.predict([[0.05], [10.05], [5]]))
    print("вероятности для x=0.05:", m.predict_proba([[0.05]]))
