import numpy as np
from sklearn.metrics import pairwise_distances


def gaussian_kernel(r):
    """Гауссово ядро K(r) = exp(-2 * r^2)"""
    return np.exp(-2 * r**2)

class ParzenKNN:
    """
    Метрический классификатор на основе метода окна Парзена переменной ширины.
    """
    def __init__(self, k=5, kernel=gaussian_kernel):
        self.k = k
        self.kernel = kernel
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes = np.unique(y)

    def predict(self, X_test):
        dists = pairwise_distances(X_test, self.X_train)
        sorted_dists = np.sort(dists, axis=1)
        
        k_idx = min(self.k, sorted_dists.shape[1] - 1)
        h = sorted_dists[:, k_idx]
        h = np.where(h == 0, 1e-10, h)
        
        r = dists / h[:, np.newaxis]
        weights = self.kernel(r)
        
        Y_onehot = (self.y_train[:, None] == self.classes).astype(float)
        class_weights = weights @ Y_onehot
        
        return self.classes[np.argmax(class_weights, axis=1)]

    def calculate_loo(self, X, y):
        """Быстрое векторизованное вычисление LOO для всей выборки"""
        dists = pairwise_distances(X, X)
        sorted_dists = np.sort(dists, axis=1)
        
        k_idx = min(self.k + 1, sorted_dists.shape[1] - 1)
        h = sorted_dists[:, k_idx]
        h = np.where(h == 0, 1e-10, h)
        
        r = dists / h[:, np.newaxis]
        weights = self.kernel(r)
        np.fill_diagonal(weights, 0)
        
        classes = np.unique(y)
        Y_onehot = (y[:, None] == classes).astype(float)
        class_weights = weights @ Y_onehot
        predictions = classes[np.argmax(class_weights, axis=1)]
        
        return np.sum(predictions != y) / len(y)

def ccv_weights(L, l, k):
    """
    Веса R(m) для CCV.

    """
    R = np.empty(k, dtype=float)
    R[0] = l / (L - 1)

    for m in range(1, k):
        R[m] = R[m - 1] * (k - m) / (L - 1 - m)

    return R


def compute_ccv(D, y, omega_idx, k_control, R):
    """
    Вычисляет CCV(Ω) для множества эталонов Ω, заданного индексами omega_idx.

    D — предвычисленная матрица расстояний между всеми объектами выборки.
    """
    n_proto = len(omega_idx)
    k = min(k_control, n_proto - 1)

    if k <= 0:
        return 0.0

    dist = D[:, omega_idx].copy()

    # Исключаем объект как собственного соседа.
    dist[omega_idx, np.arange(n_proto)] = np.inf

    # k ближайших соседей среди эталонов
    neighbors = np.argsort(dist, axis=1)[:, :k]

    # Ошибки: класс объекта != класс его m-го соседа из Ω
    errors = y[:, None] != y[omega_idx][neighbors]

    return float(np.sum(errors * R[:k]) / len(y))


def prototype_selection_greedy_removal(X, y, k_control=None, tolerance=1e-5):
    """
    Жадное удаление не-эталонов по критерию CCV.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    L = len(X)

    if k_control is None:
        k_control = max(1, int(0.2 * L))

    l = L - k_control
    R = ccv_weights(L, l, k_control)

    # Предвычисляем расстояния
    D = pairwise_distances(X, X)

    omega_idx = np.arange(L)
    current_ccv = compute_ccv(D, y, omega_idx, k_control, R)

    print(f"Initial size: {len(omega_idx)}, CCV: {current_ccv:.6f}")

    iteration = 0

    # Гарантируем, что у каждого эталона остаётся хотя бы k_control соседей
    while len(omega_idx) > k_control + 1:
        ccvs = np.empty(len(omega_idx))

        for j in range(len(omega_idx)):
            cand_idx = np.delete(omega_idx, j)
            ccvs[j] = compute_ccv(D, y, cand_idx, k_control, R)

        best_j = int(np.argmin(ccvs))

        # Удаляем, если CCV уменьшается или почти не увеличивается
        if ccvs[best_j] > current_ccv + tolerance:
            break

        omega_idx = np.delete(omega_idx, best_j)
        current_ccv = ccvs[best_j]
        iteration += 1

        if iteration % 10 == 0:
            print(
                f"Iter {iteration}: "
                f"newSize={len(omega_idx)}, "
                f"CCV={current_ccv:.6f}"
            )

    print(f"Final size: {len(omega_idx)}, CCV: {current_ccv:.6f}")

    return X[omega_idx], y[omega_idx]