"""
Задача 2: реализовать вычисление градиента функции потерь.

Подключает task01 (данные, отступ, корреляционная инициализация) -
рекуррентная цепочка "каждый файл включает предыдущий".
"""

import numpy as np
import matplotlib.pyplot as plt

from task01_margin import load_and_prepare, add_bias, correlation_init


def quadratic_loss(M):
    """Квадратичная функция потерь L(M) = (1-M)^2."""
    return (1.0 - M) ** 2


def quadratic_loss_grad_M(M):
    """dL/dM — производная потерь по отступу M."""
    return -2.0 * (1.0 - M)


def gradient(w, x, y):
    """
    Градиент потерь по вектору весов w для ОДНОГО объекта (x, y).
    Вывод по цепному правилу:
        M(w) = y*<w,x>  =>  dM/dw = y*x
        L(M) = (1-M)^2  =>  dL/dM = -2*(1-M)
        dL/dw = dL/dM * dM/dw = -2*(1-M) * y * x
    """
    M = y * np.dot(w, x) # Скалярное произведение двух векторов (или матричное умножение, если аргументы двумерные)
    return quadratic_loss_grad_M(M) * y * x


def gradient_batch(w, X, y):
    """Векторизованный градиент, усреднённый по батчу объектов."""
    M = y * (X @ w) # @ - оператор матричного умножения
    coef = quadratic_loss_grad_M(M) * y
    return (X * coef[:, None]).mean(axis=0)


def numerical_gradient(w, x, y, eps=1e-6):
    """Численная проверка градиента методом центральных разностей."""
    grad = np.zeros_like(w)
    for j in range(len(w)):
        w_plus = w.copy(); w_plus[j] += eps
        w_minus = w.copy(); w_minus[j] -= eps
        loss_plus = quadratic_loss(y * np.dot(w_plus, x))
        loss_minus = quadratic_loss(y * np.dot(w_minus, x))
        grad[j] = (loss_plus - loss_minus) / (2 * eps)
    return grad


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare()
    Xb_train = add_bias(X_train)

    w = correlation_init(Xb_train, y_train)
    x_sample, y_sample = Xb_train[0], y_train[0]

    analytic = gradient(w, x_sample, y_sample)
    numeric = numerical_gradient(w, x_sample, y_sample)
    max_abs_diff = np.max(np.abs(analytic - numeric))

    print(f"Отступ M = {y_sample * np.dot(w, x_sample):.4f}")
    print("Аналитический градиент:", np.round(analytic, 5))
    print("Численный градиент:    ", np.round(numeric, 5))
    print(f"Максимальное расхождение: {max_abs_diff:.2e}")
    print("Градиент корректен" if max_abs_diff < 1e-4 else "ОШИБКА В ГРАДИЕНТЕ")

    names = ["bias"] + feature_names
    x_pos = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x_pos - width/2, analytic, width, label="Аналитический градиент")
    ax.bar(x_pos + width/2, numeric, width, label="Численный градиент (конечные разности)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("dL/dw")
    ax.set_title("Задача 2: сверка аналитического и численного градиента")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../images/task02_gradient.png", dpi=120)
    print("\nГрафик сохранён: task02_gradient.png")
