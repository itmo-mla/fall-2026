import numpy as np


class LinearClassifier:
    def __init__(
        self,
        n_features,
        initial_weights=None
    ):
        if initial_weights is None:
            self.w = np.zeros(n_features)
        else:
            self.w = np.array(
                initial_weights,
                dtype=float
            )

    def random_initialization(self, random_state=None, scale=0.1):
        """
        Случайная инициализация весов.

        random_state — seed генератора случайных чисел.
        scale — масштаб случайных весов.
        """

        rng = np.random.default_rng(random_state)

        self.w = rng.normal(
            loc=0.0,
            scale=scale,
            size=len(self.w)
        )

        # Bias тоже пока инициализируем случайно.
        return self.w

    @staticmethod
    def fit_primary(
        X,
        y,
        n_starts=10,
        learning_rate=0.001,
        momentum=0.9,
        l2_lambda=0.001,
        n_iterations=5000,
    ):
        return LinearClassifier.multistart(
            X,
            y,
            n_starts=n_starts,
            learning_rate=learning_rate,
            momentum=momentum,
            l2_lambda=l2_lambda,
            n_iterations=n_iterations,
        )

    @staticmethod
    def multistart(
            X,
            y,
            n_starts=10,
            learning_rate=0.001,
            momentum=0.9,
            l2_lambda=0.001,
            n_iterations=5000
    ):
        """
        Multistart:

        1. Создаём несколько моделей
        со случайными начальными весами.
        2. Обучаем каждую.
        3. Считаем итоговый train Loss.
        4. Выбираем модель с минимальным Loss.
        """

        best_model = None
        best_loss = float("inf")
        best_history = []
        results = []

        for start in range(n_starts):

            rng = np.random.default_rng(start)

            initial_weights = rng.normal(
                0.0,
                0.1,
                size=X.shape[1]
            )

            model = LinearClassifier(
                X.shape[1],
                initial_weights=initial_weights
            )

            loss_history = model.fit_sgd_momentum_l2(
                X,
                y,
                learning_rate=learning_rate,
                momentum=momentum,
                l2_lambda=l2_lambda,
                n_iterations=n_iterations,
                random_state=start
            )

            final_loss = model.loss(X, y)

            results.append(final_loss)

            print(
                f"Запуск {start + 1}: "
                f"начальный Loss = {loss_history[0]:.6f}, "
                f"конечный Loss = {final_loss:.6f}"
            )

            if final_loss < best_loss:
                best_loss = final_loss
                best_model = model
                best_history = loss_history

        return best_model, results, best_history

    @staticmethod
    def correlation_initialization(X, y):

        n_features = X.shape[1]

        w = np.zeros(n_features)

        # Первый столбец — bias.
        # Начинаем с нуля.
        w[0] = 0.0

        # Остальные признаки
        for j in range(1, n_features):

            feature = X[:, j]

            correlation = np.corrcoef(
                feature,
                y
            )[0, 1]

            w[j] = correlation

        return w

    def predict_score(self, X):
        """
        f(x) = <x, w>
        """
        return X @ self.w

    def predict(self, X):
        """
        f(x) >= 0 -> +1
        f(x) < 0  -> -1
        """
        scores = self.predict_score(X)

        return np.where(scores >= 0, 1, -1)

    def margin(self, X, y):
        """
        M_i = y_i * <x_i, w>
        """
        return y * self.predict_score(X)

    def loss(self, X, y):
        """
        Q(w) = 1/(2n) * sum((y_i - <x_i, w>)^2)
        """
        predictions = self.predict_score(X)

        errors = y - predictions

        return np.mean(errors ** 2) / 2

    

    def gradient(self, X, y):
        """
        grad Q(w) = 1/n * X^T * (Xw - y)
        """
        n = X.shape[0]

        predictions = self.predict_score(X)

        return X.T @ (predictions - y) / n

    def stochastic_gradient(self, x, y):
        """
        Q_i(w) = 1/2 * (y_i - <x_i, w>)^2

        grad Q_i(w) = x_i * (<x_i, w> - y_i)
        """

        prediction = x @ self.w

        return x * (prediction - y)

    def stochastic_gradient_l2(self, x, y, l2_lambda):
        """
        grad = x * (<x, w> - y) + lambda * w
        """

        prediction = x @ self.w

        gradient = x * (prediction - y)

        regularization = l2_lambda * self.w.copy()

        # Bias не регуляризуем
        regularization[0] = 0

        return gradient + regularization


    def fit(self, X, y, learning_rate=0.01, n_iterations=1000):

        loss_history = []

        for _ in range(n_iterations):
            gradient = self.gradient(X, y)

            self.w -= learning_rate * gradient

            loss = self.loss(X, y)

            loss_history.append(loss)

        return loss_history

    def fit_sgd_momentum(
        self,
        X,
        y,
        learning_rate=0.01,
        momentum=0.9,
        n_iterations=1000,
        random_state=42
    ):

        rng = np.random.default_rng(random_state)

        velocity = np.zeros_like(self.w)

        loss_history = []

        n_samples = X.shape[0]

        for _ in range(n_iterations):

            # Случайно выбираем один 
            index = rng.integers(0, n_samples)

            x_i = X[index]
            y_i = y[index]

            gradient = self.stochastic_gradient(x_i, y_i)

            velocity = momentum * velocity + gradient

            self.w -= learning_rate * velocity

            loss = self.loss(X, y)

            loss_history.append(loss)

        return loss_history

    def fit_sgd_momentum_l2(
        self,
        X,
        y,
        learning_rate=0.01,
        momentum=0.9,
        l2_lambda=0.001,
        n_iterations=1000,
        random_state=42
    ):

        rng = np.random.default_rng(random_state)

        velocity = np.zeros_like(self.w)

        loss_history = []

        n_samples = X.shape[0]

        for _ in range(n_iterations):

            index = rng.integers(0, n_samples)

            x_i = X[index]
            y_i = y[index]

            gradient = self.stochastic_gradient_l2(
                x_i,
                y_i,
                l2_lambda
            )

            velocity = (
                momentum * velocity
                + gradient
            )

            self.w -= learning_rate * velocity

            # Значение функции потерь на всей обучающей выборке
            loss_history.append(
                self.loss(X, y)
            )

        return loss_history

    def fit_steepest_descent(
        self,
        X,
        y,
        n_iterations=100
    ):

        loss_history = []

        n = X.shape[0]

        for _ in range(n_iterations):

            gradient = self.gradient(X, y)

            X_gradient = X @ gradient

            numerator = gradient @ gradient

            denominator = (
                X_gradient @ X_gradient
            ) / n

            if denominator < 1e-12:
                break

            alpha = numerator / denominator

            self.w -= alpha * gradient

            loss_history.append(
                self.loss(X, y)
            )

        return loss_history

    def fit_sgd_margin(
        self,
        X,
        y,
        learning_rate=0.01,
        n_epochs=10
    ):
        loss_history = []

        n_samples = X.shape[0]

        for epoch in range(n_epochs):

            # Текущие отступы
            margins = self.margin(X, y)

            # Порядок объектов по |M_i|
            indices = np.argsort(np.abs(margins))


            for index in indices:

                x_i = X[index]
                y_i = y[index]

                gradient = self.stochastic_gradient(
                    x_i,
                    y_i
                )

                self.w -= learning_rate * gradient

            loss_history.append(
                self.loss(X, y)
            )

        return loss_history

    def fit_sgd_recursive_quality(
        self,
        X,
        y,
        learning_rate=0.01,
        n_iterations=1000,
        lambda_quality=0.1,
        random_state=42
    ):
        """
        SGD с рекуррентной оценкой качества.

        Q_t = lambda * xi_t + (1 - lambda) * Q_{t-1}
        """

        rng = np.random.default_rng(random_state)

        n_samples = X.shape[0]

        # Начальная оценка качества
        quality = 0.0

        quality_history = []
        loss_history = []

        for _ in range(n_iterations):

            index = rng.integers(0, n_samples)

            x_i = X[index]
            y_i = y[index]

            prediction = x_i @ self.w

            # Ошибка на одном объекте
            xi = 0.5 * (y_i - prediction) ** 2

            # Стохастический градиент
            gradient = x_i * (prediction - y_i)

            self.w -= learning_rate * gradient

            quality = (
                lambda_quality * xi
                + (1 - lambda_quality) * quality
            )

            quality_history.append(quality)
            loss_history.append(self.loss(X, y))

        self.quality_history = quality_history
        return loss_history

    def fit_sgd_random_presentation(
        self,
        X,
        y,
        learning_rate=0.001,
        n_epochs=20,
        random_state=42
    ):

        rng = np.random.default_rng(random_state)

        n_samples = X.shape[0]
        loss_history = []

        for epoch in range(n_epochs):

            indices = rng.permutation(n_samples)

            for index in indices:

                x_i = X[index]
                y_i = y[index]

                gradient = self.stochastic_gradient(
                    x_i,
                    y_i
                )

                self.w -= learning_rate * gradient

            loss_history.append(
                self.loss(X, y)
            )

        return loss_history

    def sort_by_margin(self, X, y):
        
        margins = self.margin(X, y)

        indices = np.argsort(np.abs(margins))

        return indices
    