import numpy as np


class LogisticRegressionIRLS:
    """
    Логистическая регрессия, реализованная через метод Ньютона-Рафсона (IRLS).
    
    Согласно лекции, логистическая регрессия является частным случаем GLM,
    где целевая переменная имеет распределение Бернулли, а функция связи - сигмоида.
    Метод Ньютона-Рафсона сводит задачу к итеративному решению взвешенной 
    линейной регрессии (IRLS - Iteratively Reweighted Least Squares).
    """
    
    def __init__(self, max_iter=100, tol=1e-6, C=1.0):
        """
        :param max_iter: Максимальное количество итераций IRLS
        :param tol: Критерий остановки по норме изменения весов
        :param C: Обратный коэффициент регуляризации (как в sklearn). lambda = 1 / C
        """
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
    
    def fit(self, X, y):
        """
        Обучение модели методом Ньютона-Рафсона (IRLS).
        
        На каждой итерации:
        1. Вычисляются вероятности mu_i = sigma(x_i^T w)
        2. Формируется диагональная матрица весов W = diag(mu_i * (1 - mu_i))
        3. Обновляются веса: w_{t+1} = w_t - H^{-1} * grad, где H = X^T W X + lambda*I
        """
        n_samples, n_features = X.shape
        
        # Добавляем столбец единиц для учета свободного члена (intercept)
        X_b = np.c_[np.ones((n_samples, 1)), X]
        n_features_b = n_features + 1
        
        # Начальное приближение весов (нулевое)
        w = np.zeros(n_features_b)
        lambda_reg = 1.0 / self.C
        
        # Вектор регуляризации (не регуляризуем свободный член)
        reg_vector = lambda_reg * np.ones(n_features_b)
        reg_vector[0] = 0.0
        
        for iteration in range(self.max_iter):
            # Линейная комбинация признаков: z = X^T w
            z = X_b @ w
            
            # Сигмоидная функция с обрезкой для численной стабильности
            mu = 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
            
            # Градиент функционала: grad = X^T (mu - y) + lambda * w
            grad = X_b.T @ (mu - y) + reg_vector * w
            
            # Диагональная матрица весов W = diag(mu * (1 - mu))
            # Чем ближе mu_i к 0.5, тем больше вес объекта
            W_diag = mu * (1 - mu)
            
            # Гессиан: H = X^T W X + lambda * I
            hess = X_b.T @ (W_diag[:, np.newaxis] * X_b) + np.diag(reg_vector)
            
            # Добавляем малое число для обеспечения положительной определенности гессиана
            hess += 1e-8 * np.eye(n_features_b)
            
            # Решение системы линейных уравнений для шага Ньютона: H * delta = grad
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Fallback на псевдообратную матрицу в случае вырожденности
                delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
            
            # Обновление весов: w_{t+1} = w_t - delta
            w = w - delta
            self.n_iter_ = iteration + 1
            
            # Проверка критерия остановки
            if np.linalg.norm(delta) < self.tol:
                break
        
        # Разделяем веса на коэффициенты и свободный член
        self.intercept_ = w[0]
        self.coef_ = w[1:]
        
        return self
    
    def predict_proba(self, X):
        """
        Вычисление вероятностей принадлежности к классам.
        
        :param X: Матрица признаков
        :return: Массив вероятностей [P(y=0), P(y=1)]
        """
        z = X @ self.coef_ + self.intercept_
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
        return np.c_[1 - mu, mu]
    
    def predict(self, X):
        """
        Предсказание меток классов.
        
        :param X: Матрица признаков
        :return: Массив предсказанных меток (0 или 1)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)