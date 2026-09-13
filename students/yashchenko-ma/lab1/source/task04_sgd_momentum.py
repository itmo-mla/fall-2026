"""
Задача 4: реализовать метод стохастического градиентного спуска с инерцией.

Подключает task03 (and task02, task01).

Класс LinearClassifierMomentum - база всей дальнейшей цепочки классов:
    LinearClassifierMomentum (задача 4)
        -> LinearClassifierL2          (задача 5, task05_l2.py)
            -> LinearClassifierSteepest (задача 6, task06_steepest.py)
                -> LinearClassifier     (задача 7, task07_margin_presentation.py)

Каждый следующий класс переопределяет ровно один "крючок" метода fit(),
не трогая остальную логику - так реализации задач 5-7 физически не могут
случайно сломать уже проверенную логику задачи 4.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from task03_quality import (load_and_prepare, add_bias, correlation_init,
                             quadratic_loss, init_Q, update_Q)
from task02_gradient import gradient
from task01_margin import plot_margin_distribution


class LinearClassifierMomentum:
    """SGD с инерцией (momentum) на квадратичной функции потерь."""

    def __init__(self, eta=0.01, gamma=0.9, n_epochs=10, lam_Q=None, random_state=None):
        self.eta = eta
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.lam_Q = lam_Q
        self.random_state = random_state
        self.w = None
        self.Q_history = []

    # ---- точки расширения для последующих задач (по умолчанию — no-op) ----
    def _regularize(self, grad, w):
        """Задача 5 (L2) переопределяет этот метод."""
        return grad

    def _compute_step(self, x_i):
        """Задача 6 (скорейший спуск) переопределяет этот метод."""
        return self.eta

    def _get_order(self, rng, n, X, y):
        """Задача 7 (предъявление по модулю отступа) переопределяет этот метод."""
        return rng.permutation(n)
    # -------------------------------------------------------------------

    def fit(self, X, y, w_init=None):
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)
        self.w = w_init.copy() if w_init is not None else np.zeros(d)
        v = np.zeros(d)

        lam = self.lam_Q if self.lam_Q is not None else 1.0 / n
        Q = init_Q(self.w, X, y, sample_size=min(50, n), random_state=self.random_state)
        self.Q_history = [Q]

        for epoch in range(self.n_epochs):
            order = self._get_order(rng, n, X, y)
            for i in order:
                x_i, y_i = X[i], y[i]

                grad_i = gradient(self.w, x_i, y_i)
                grad_i = self._regularize(grad_i, self.w)
                step = self._compute_step(x_i)

                v = self.gamma * v + (1 - self.gamma) * grad_i
                self.w = self.w - step * v

                M_i = y_i * np.dot(self.w, x_i)
                Q = update_Q(Q, quadratic_loss(M_i), lam)
                self.Q_history.append(Q)
        return self

    def decision_function(self, X):
        return X @ self.w

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare()
    Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

    clf = LinearClassifierMomentum(eta=0.01, gamma=0.9, n_epochs=10, random_state=42)
    clf.fit(Xb_train, y_train)

    y_pred_test = clf.predict(Xb_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(classification_report(y_test, y_pred_test,
                                 target_names=["No failure", "Failure"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(clf.Q_history, linewidth=0.7, label="Q (рекуррентная оценка)")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Q")
    ax.set_title("Задача 4: сходимость SGD с инерцией")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../images/task04_sgd_momentum.png", dpi=120)
    print("\nГрафик сохранён: task04_sgd_momentum.png")

    print("\nВажное наблюдение: модель «схлопывается» в мажоритарный класс "
          "(recall=0 на отказах) - квадратичная потеря без взвешивания классов "
          "при дисбалансе 96.6%/3.4% сходится именно так. Это будет частично "
          "исправлено в задаче 7 (предъявление по модулю отступа).")

    # Отступ ПОСЛЕ обучения (на train) - для сравнения с "до обучения" из task01
    # (random.png / correlation.png): доля M<0 должна заметно снизиться.
    plot_margin_distribution(clf.w, Xb_train, y_train,
                              "Задача 4: отступы объектов после обучения (SGD с инерцией)",
                              "task04_pretrained.png")