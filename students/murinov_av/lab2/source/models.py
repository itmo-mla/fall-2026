import numpy as np
from scipy.special import comb
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

def compute_ccv(X_total, y_total, Omega_X, Omega_y, k_control, R):
    """
    Вычисляет CCV (Complete Cross-Validation) для множества эталонов Omega.
    
    CCV(Ω) = 1/L * sum_{i=1}^L sum_{m=1}^k [y_i != y_i^(m|Ω)] * R(m)
    
    где:
    - L = len(X_total) - полный размер выборки
    - Ω = (Omega_X, Omega_y) - множество эталонов
    - y_i^(m|Ω) - класс m-го ближайшего соседа объекта x_i среди Ω
    - k = k_control - длина контроля
    - R(m) = C^{L-1-m}_{l-1} / C^{L-1}_{l} - веса (l = L - k)
    """
    L = len(X_total)
    if len(Omega_X) == 0:
        return float('inf')
    
    # Матрица расстояний от всех объектов X_total до эталонов Omega_X
    dists = pairwise_distances(X_total, Omega_X)
    
    # Находим k_control ближайших соседей в Omega для каждого объекта из X_total
    # Используем argpartition для эффективности (быстрее полной сортировки)
    if len(Omega_X) <= k_control:
        neighbor_indices = np.argsort(dists, axis=1)[:, :len(Omega_X)]
        actual_k = len(Omega_X)
    else:
        neighbor_indices = np.argpartition(dists, k_control, axis=1)[:, :k_control]
        # Сортируем только k ближайших для правильного порядка
        for i in range(L):
            neighbor_indices[i] = neighbor_indices[i][np.argsort(dists[i, neighbor_indices[i]])]
        actual_k = k_control
    
    # Вычисляем CCV
    ccv_sum = 0.0
    for m in range(actual_k):
        # m-й сосед (0-indexed, т.е. (m+1)-й в 1-indexed)
        # Класс m-го соседа для каждого объекта
        neighbor_classes = Omega_y[neighbor_indices[:, m]]
        # Ошибки: y_i != neighbor_classes
        errors = (y_total != neighbor_classes).astype(float)
        ccv_sum += np.sum(errors) * R[m]
    
    return ccv_sum / L

def prototype_selection_greedy_removal(X, y, k_control, tolerance=1e-5):
    """
    Жадное удаление не-эталонов по критерию CCV.
    """
    X = np.array(X)
    y = np.array(y)
    L = len(X)
        
    l = L - k_control
    
    R = np.zeros(k_control)
    den = comb(L - 1, l, exact=True)
    for m in range(1, k_control + 1):
        num = comb(L - 1 - m, l - 1, exact=True)
        R[m - 1] = num / den
    
    Omega_X = X.copy()
    Omega_y = y.copy()
    
    current_ccv = compute_ccv(X, y, Omega_X, Omega_y, k_control, R)
    print(f"\nInitial size: {len(Omega_X)}, CCV: {current_ccv:.6f}")
    
    improved = True
    iteration = 0
    
    while improved and len(Omega_X) > 10:
        improved = False
        best_ccv = current_ccv
        best_idx_to_remove = -1
        
        for i in range(len(Omega_X)):
            temp_X = np.delete(Omega_X, i, axis=0)
            temp_y = np.delete(Omega_y, i)
            temp_ccv = compute_ccv(X, y, temp_X, temp_y, k_control, R)
            
            if temp_ccv <= best_ccv + tolerance:
                best_ccv = temp_ccv
                best_idx_to_remove = i
        
        if best_idx_to_remove != -1:
            Omega_X = np.delete(Omega_X, best_idx_to_remove, axis=0)
            Omega_y = np.delete(Omega_y, best_idx_to_remove)
            current_ccv = best_ccv
            improved = True
            iteration += 1
            if iteration % 10 == 0:
                print(f"Iter {iteration}: newSize={len(Omega_X)}, CCV={current_ccv:.6f}")
    
    print(f"Final size: {len(Omega_X)}, CCV: {current_ccv:.6f}")
    return Omega_X, Omega_y