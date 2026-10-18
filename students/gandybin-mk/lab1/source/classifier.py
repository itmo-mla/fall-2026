import numpy as np


class LinearClassifier:
    def __init__(
        self,
        learning_rate=0.01,
        beta=0.9,
        l2=0.01,
        alpha=0.01,
        epochs=100,
        random_state=42,
        optimizer="momentum",
        init_method="random",
        sample_order="random",
        bias=0.0,
    ):
        """Задает гиперпараметры, режимы обучения и начальное состояние модели.

        learning_rate — постоянный шаг SGD; beta — коэффициент инерции.
        l2 — коэффициент L2-регуляризации; alpha — коэффициент сглаживания Q.
        epochs — число эпох SGD или предел итераций скорейшего спуска.
        random_state — начальное состояние генератора случайных чисел.
        optimizer — "momentum" или "steepest".
        init_method — "random" или "correlation".
        sample_order — "random" или "margin".
        bias — начальное значение свободного члена.

        w и b хранят параметры модели; v и v_b — накопленные градиенты.
        Q и quality_history хранят рекуррентную оценку качества и ее историю.
        Генератор rng создается при инициализации весов.
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.l2 = l2
        self.alpha = alpha
        self.epochs = epochs

        self.random_state = random_state
        self.optimizer = optimizer
        self.init_method = init_method
        self.sample_order = sample_order

        self.initial_bias = bias

        self.w = None
        self.b = bias

        self.v = None
        self.v_b = 0.0

        self.Q = None
        self.quality_history = []

        self.rng = None

    def _initialize_weights(self, X, y):
        """Инициализирует веса и генератор случайных чисел перед обучением."""
        n_features = X.shape[1]

        self.rng = np.random.default_rng(self.random_state)

        if self.init_method == "random":
            self.w = self.rng.normal(
                loc=0.0,
                scale=0.01,
                size=n_features,
            )

        elif self.init_method == "correlation":
            self.w = np.array(
                [np.corrcoef(X[:, j], y)[0, 1] for j in range(n_features)]
            )

            self.w = np.nan_to_num(self.w)

        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        self.v = np.zeros_like(self.w)
        self.v_b = 0.0

        self.b = self.initial_bias

    def forward(self, X):
        """Вычисляет линейную оценку X @ w + b для объекта или выборки."""
        return X @ self.w + self.b

    def margin(self, X, y):
        """Вычисляет отступы y * f(X).

        Положительный отступ соответствует правильной классификации,
        отрицательный — ошибке, нулевой — границе решения.
        """
        return y * self.forward(X)

    def loss(self, X, y):
        """Возвращает квадратичные потери (f(X) - y) ** 2."""
        prediction = self.forward(X)
        return (prediction - y) ** 2

    def mean_loss(self, X, y, regularized=False):
        """Вычисляет среднюю квадратичную потерю на выборке."""
        risk = np.mean(self.loss(X, y))

        if regularized:
            risk += self.l2 * np.sum(self.w**2)

        return float(risk)

    def _get_sample_order(self, X, y):
        """Возвращает индексы объектов в порядке предъявления на текущую эпоху."""
        n_samples = X.shape[0]

        if self.sample_order == "random":
            return self.rng.permutation(n_samples)

        elif self.sample_order == "margin":
            margins = self.margin(X, y)

            return np.argsort(np.abs(margins))

        else:
            raise ValueError(f"Unknown sample_order: {self.sample_order}")

    def gradient(self, x, y):
        """Вычисляет градиент квадратичной потери одного объекта без L2-штрафа."""
        prediction = self.forward(x)

        error = prediction - y

        grad_w = 2 * error * x
        grad_b = 2 * error

        return grad_w, grad_b

    def _l2_gradient(self):
        """Вычисляет градиент L2-штрафа по весам: 2 * l2 * w."""
        return 2 * self.l2 * self.w

    def _update_momentum(self, grad_w, grad_b):
        """Обновляет накопленные градиенты весов и смещения с инерцией."""
        self.v = self.beta * self.v + grad_w

        self.v_b = self.beta * self.v_b + grad_b

        return self.v, self.v_b

    def _update_weights(self, grad_w, grad_b):
        """Выполняет шаг SGD с инерцией для весов и свободного члена."""
        v_w, v_b = self._update_momentum(
            grad_w,
            grad_b,
        )

        self.w -= self.learning_rate * v_w
        self.b -= self.learning_rate * v_b

    def _update_quality(self, current_loss):
        """Обновляет рекуррентную оценку качества и сохраняет ее в истории."""
        if self.Q is None:
            self.Q = float(current_loss)

        else:
            self.Q = (1 - self.alpha) * self.Q + self.alpha * current_loss

        self.quality_history.append(self.Q)

        return self.Q

    def _full_gradient(self, X, y):
        """Вычисляет полный градиент регуляризованного риска на всей выборке."""
        predictions = self.forward(X)

        errors = predictions - y

        grad_w = 2 * X.T @ errors / len(y) + self._l2_gradient()

        grad_b = 2 * np.mean(errors)

        return grad_w, grad_b

    def _find_best_step(
        self,
        X,
        y,
        grad_w,
        grad_b,
    ):
        """Находит шаг, минимизирующий квадратичный риск вдоль антиградиента."""

        residuals = self.forward(X) - y

        projection = X @ grad_w + grad_b

        numerator = np.mean(projection * residuals) + self.l2 * np.dot(self.w, grad_w)

        denominator = np.mean(projection**2) + self.l2 * np.dot(grad_w, grad_w)

        if denominator <= 1e-12:
            return 0.0

        eta = numerator / denominator

        return max(0.0, float(eta))

    def fit(self, X, y):
        """Обучает классификатор.

        В режиме "momentum" каждую эпоху определяет порядок объектов.
        Для каждого объекта вычисляет потерю с L2-штрафом и обновляет Q,
        затем добавляет L2 к градиенту потери и выполняет шаг SGD с инерцией.

        В режиме "steepest" обновляет Q по полному регуляризованному риску,
        вычисляет полный градиент и оптимальный шаг, затем обновляет параметры.
        Обучение завершается при нулевом шаге или достижении предела epochs.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._initialize_weights(X, y)

        self.Q = None
        self.quality_history = []

        if self.optimizer == "momentum":
            for epoch in range(self.epochs):
                indices = self._get_sample_order(X, y)

                for idx in indices:
                    x_i = X[idx]
                    y_i = y[idx]

                    current_loss = self.loss(
                        x_i,
                        y_i,
                    ) + self.l2 * np.sum(self.w**2)

                    self._update_quality(current_loss)

                    grad_w, grad_b = self.gradient(
                        x_i,
                        y_i,
                    )

                    grad_w += self._l2_gradient()

                    self._update_weights(
                        grad_w,
                        grad_b,
                    )

        elif self.optimizer == "steepest":
            for epoch in range(self.epochs):
                current_loss = self.mean_loss(
                    X,
                    y,
                    regularized=True,
                )

                self._update_quality(current_loss)

                grad_w, grad_b = self._full_gradient(
                    X,
                    y,
                )

                eta = self._find_best_step(
                    X,
                    y,
                    grad_w,
                    grad_b,
                )

                self.w -= eta * grad_w
                self.b -= eta * grad_b

                if eta == 0:
                    break

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        return self

    def predict(self, X):
        """Возвращает метки классов по знаку линейной оценки.

        При f(X) >= 0 выбирает +1, иначе -1.
        """
        scores = self.forward(X)

        return np.where(
            scores >= 0,
            1,
            -1,
        )
