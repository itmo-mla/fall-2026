"""Метрические методы классификации (лекция 2).

Обозначения 1-в-1 с лекцией::

    X --- объекты, Y --- ответы;
    X^l = (x_i, y_i)_{i=1}^l --- обучающая выборка;
    x = (x^1, ..., x^n) --- вектор признаков объекта x (ВЕРХНИЕ индексы);
    rho(x, x_i) = (Σ_j w_j |x^j − x^j_i|^p)^{1/p} --- метрика Минковского;
    rho(x, x^{(1)}) ≤ ... ≤ rho(x, x^{(l)}) --- ранжировка соседей;
    a(x; X^l) = argmax_{y ∈ Y} Σ_i [y^{(i)} = y] · w(i, x) --- обобщённый
        метрический классификатор;
    Gamma_y(x) --- мера голосов соседей в пользу класса y;
    w(i, x) = [i ≤ 1] --- 1NN, [i ≤ k] --- kNN, [i ≤ k] w_i --- взвешенный kNN;
    LOO(k, X^l) = Σ_i [a(x_i; X^l \\ {x_i}, k) ≠ y_i] → min_k --- СУММА ошибок;
    w(i, x) = K(rho(x, x^{(i)}) / h) --- окно Парзена, h --- ширина окна;
    a(x; X^l, k, K) = argmax_{y ∈ Y} Σ_i [y_i = y] · K(rho(x, x_i) / rho(x, x^{(k+1)})).

По умолчанию w_j = 1, p = 2 (обычная евклидова метрика).
"""

import numpy as np


def rho(x, x_i, w=None, p=2):
    """Расстояние Минковского: rho(x, x_i) = (Σ_j w_j |x^j − x^j_i|^p)^{1/p}.

    w --- веса признаков w_j (по умолчанию все w_j = 1), p --- степень (по
    умолчанию p = 2, евклидова метрика).
    """
    x = np.asarray(x, dtype=float)
    x_i = np.asarray(x_i, dtype=float)
    dist = np.abs(x - x_i) ** p
    if w is not None:
        dist = np.asarray(w, dtype=float) * dist
    return float(np.sum(dist) ** (1.0 / p))


def rho_matrix(X, Z, w=None, p=2):
    """Матрица попарных расстояний rho между строками X и строками Z."""
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    dist = np.abs(X[:, None, :] - Z[None, :, :]) ** p
    if w is not None:
        dist = dist * np.asarray(w, dtype=float)
    return np.sum(dist, axis=2) ** (1.0 / p)


# Ядра K(r) из лекции.
def Pi(r):
    """Прямоугольное ядро: Π(r) = [|r| ≤ 1]."""
    r = np.asarray(r, dtype=float)
    return (np.abs(r) <= 1).astype(float)


def T(r):
    """Треугольное ядро: T(r) = (1 − |r|) [|r| ≤ 1]."""
    r = np.asarray(r, dtype=float)
    a = np.abs(r)
    return ((1.0 - a) * (a <= 1)).astype(float)


def E(r):
    """Квадратичное ядро Епанечникова: E(r) = (1 − r^2) [|r| ≤ 1]."""
    r = np.asarray(r, dtype=float)
    return ((1.0 - r * r) * (np.abs(r) <= 1)).astype(float)


def Q(r):
    """Квартическое ядро: Q(r) = (1 − r^2)^2 [|r| ≤ 1]."""
    r = np.asarray(r, dtype=float)
    return ((1.0 - r * r) ** 2 * (np.abs(r) <= 1)).astype(float)


def G(r):
    """Гауссовское ядро: G(r) = exp(−2 r^2)."""
    r = np.asarray(r, dtype=float)
    return np.exp(-2.0 * r * r)


def Gamma_y_from_distances(distances, y, k, K=G):
    """Мера голосов Gamma_y(x) по готовому вектору расстояний rho(x, x_i).

    Окно Парзена переменной ширины: h = rho(x, x^{(k+1)}), сумма
    НЕранжированная: Gamma_y(x) = Σ_i [y_i = y] · K(rho(x, x_i) / h).
    Возвращает dict {y: Gamma_y(x)} (ключи отсортированы по возрастанию).
    """
    distances = np.asarray(distances, dtype=float)
    y = np.asarray(y)
    order = np.argsort(distances, kind="stable")
    h = float(distances[order[min(k, len(distances) - 1)]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = distances / h
    ratios = np.where(distances == 0.0, 0.0, ratios)  # соглашение 0/0 = 0
    weights = np.asarray(K(ratios), dtype=float)
    Gamma_y = {}
    for label in sorted(np.unique(y).tolist()):
        Gamma_y[label] = float(weights[y == label].sum())
    return Gamma_y


def Gamma_y(x, X_l, y, k, K=G, w=None, p=2):
    """Мера голосов Gamma_y(x) = Σ_i [y_i = y] · K(rho(x, x_i) / h).

    X_l --- матрица признаков обучающей выборки X^l, y --- ответы,
    h = rho(x, x^{(k+1)}).
    """
    distances = rho_matrix(np.asarray(x, dtype=float)[None, :],
                           np.asarray(X_l, dtype=float), w, p)[0]
    return Gamma_y_from_distances(distances, y, k, K)


def a_from_distances(distances, y, k, K=G):
    """Классификатор a(x; X^l, k, K) по вектору расстояний (argmax Gamma_y)."""
    Gamma = Gamma_y_from_distances(distances, y, k, K)
    return max(sorted(Gamma), key=lambda label: Gamma[label])


def a(x, X_l, y, k, K=G, w=None, p=2):
    """Классификатор Парзена переменной ширины a(x; X^l, k, K).

    a(x; X^l, k, K) = argmax_{y ∈ Y} Σ_i [y_i = y] · K(rho(x, x_i) / h),
    h = rho(x, x^{(k+1)}). При равных Gamma_y побеждает меньший класс.
    """
    Gamma = Gamma_y(x, X_l, y, k, K, w, p)
    return max(sorted(Gamma), key=lambda label: Gamma[label])


class ParzenKNN:
    """Окно Парзена переменной ширины a(x; X^l, k, K) (векторный интерфейс)."""

    def __init__(self, k=5, K=G, w=None, p=2):
        self.k = k
        self.K = K
        self.w = w
        self.p = p

    def fit(self, X_l, y):
        self.X_l = np.asarray(X_l, dtype=float)
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        D = rho_matrix(X, self.X_l, self.w, self.p)
        return np.array([a_from_distances(row, self.y, self.k, self.K)
                         for row in D])


def LOO(k, X_l, y, K=G, w=None, p=2):
    """Скользящий контроль: LOO(k, X^l) = Σ_i [a(x_i; X^l \\ {x_i}, k) ≠ y_i].

    СУММА ошибок (без 1/l). Классификатор a(…; k) принимает параметр k.
    """
    X_l = np.asarray(X_l, dtype=float)
    y = np.asarray(y)
    D = rho_matrix(X_l, X_l, w, p)
    errors = 0
    for i in range(len(X_l)):
        mask = np.ones(len(X_l), dtype=bool)
        mask[i] = False
        errors += a_from_distances(D[i][mask], y[mask], k, K) != y[i]
    return errors


def select_k_by_LOO(X_l, y, k_values, K=G, w=None, p=2):
    """Подбор k: LOO(k, X^l) → min_k. Возвращает (k*, [LOO(k, X^l)])."""
    LOOs = [LOO(k, X_l, y, K, w, p) for k in k_values]
    return int(k_values[int(np.argmin(LOOs))]), LOOs
