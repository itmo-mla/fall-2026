"""
Задача 6: реализовать скорейший градиентный спуск.

Подключает task05 (and task04, task03, task02, task01).
Переопределяет только _compute_step().
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from task05_l2 import LinearClassifierL2, load_and_prepare, add_bias


class LinearClassifierSteepest(LinearClassifierL2):
    """SGD с инерцией + L2 + опциональный скорейший градиентный спуск."""

    def __init__(self, step_mode="fixed", **kwargs):
        super().__init__(**kwargs)
        self.step_mode = step_mode  # 'fixed' -> self.eta; 'steepest' -> адаптивный шаг

    def _compute_step(self, x_i):
        if self.step_mode == "steepest":
            # Оптимальный шаг вдоль направления градиента для квадратичной
            # потери на объекте i: h* = 1 / (2*||x_i||^2).
            return 1.0 / (2.0 * np.dot(x_i, x_i) + 1e-12)
        return self.eta


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare()
    Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

    clf_fixed = LinearClassifierSteepest(eta=0.01, gamma=0.9, n_epochs=5,
                                          step_mode="fixed", random_state=42)
    clf_fixed.fit(Xb_train, y_train)

    clf_steep = LinearClassifierSteepest(gamma=0.9, n_epochs=5,
                                          step_mode="steepest", random_state=42)
    clf_steep.fit(Xb_train, y_train)

    print(f"Fixed eta:  Q_final={clf_fixed.Q_history[-1]:.4f}  "
          f"test_acc={accuracy_score(y_test, clf_fixed.predict(Xb_test)):.4f}")
    print(f"Steepest:   Q_final={clf_steep.Q_history[-1]:.4f}  "
          f"test_acc={accuracy_score(y_test, clf_steep.predict(Xb_test)):.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(clf_fixed.Q_history, label="Фиксированный eta=0.01", linewidth=0.8)
    ax.plot(clf_steep.Q_history, label="Скорейший спуск", linewidth=0.8)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Q")
    ax.set_title("Задача 6: фиксированный шаг vs скорейший градиентный спуск")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../images/task06_steepest.png", dpi=120)
    print("\nГрафик сохранён: task06_steepest.png")

    print("\nАнализ: скорейший спуск нестабилен в связке с инерцией - h* "
          "выведен для чистого градиента объекта, но применяется к "
          "сглаженному моментум-вектору, что ломает точность аналитической "
          "минимизации.")
