"""
Задача 3: реализовать рекуррентную оценку функционала качества.

Подключает task02 (and via - task01).
"""

import numpy as np
import matplotlib.pyplot as plt

from task02_gradient import load_and_prepare, add_bias, correlation_init, quadratic_loss


def empirical_risk(w, X, y):
    """Точный эмпирический риск (для сверки, не для внутреннего цикла SGD)."""
    M = y * (X @ w)
    return quadratic_loss(M).mean()


def init_Q(w, X, y, sample_size=None, random_state=None):
    """Инициализация Q по случайной подвыборке перед началом SGD."""
    n = X.shape[0]
    if sample_size is None:
        sample_size = min(50, n)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=sample_size, replace=False)
    M = y[idx] * (X[idx] @ w)
    return quadratic_loss(M).mean()


def update_Q(Q, loss_i, lam):
    """Рекуррентное обновление: Q := lambda*L_i + (1-lambda)*Q."""
    return lam * loss_i + (1 - lam) * Q


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare()
    Xb_train = add_bias(X_train)

    w = correlation_init(Xb_train, y_train)
    n = Xb_train.shape[0]
    true_Q = empirical_risk(w, Xb_train, y_train)
    print(f"Истинный эмпирический риск (по всей выборке): {true_Q:.4f}")

    lam = 1.0 / n
    rng = np.random.RandomState(0)
    order = rng.permutation(n)

    Q = init_Q(w, Xb_train, y_train, sample_size=50, random_state=0)
    Q_history = [Q]
    for i in order:
        M_i = y_train[i] * np.dot(w, Xb_train[i])
        Q = update_Q(Q, quadratic_loss(M_i), lam)
        Q_history.append(Q)

    print(f"Рекуррентная оценка Q после одного прохода: {Q_history[-1]:.4f}")
    print(f"Расхождение с истинным риском: {abs(Q_history[-1] - true_Q):.5f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(Q_history, label="Рекуррентная оценка Q", linewidth=1)
    ax.axhline(true_Q, color="red", linestyle="--", label="Истинный эмпирический риск")
    ax.set_xlabel("Номер предъявленного объекта")
    ax.set_ylabel("Q")
    ax.set_title(f"Задача 3: сходимость рекуррентной оценки Q (λ=1/n={lam:.2e})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../images/task03_quality.png", dpi=120)
    print("\nГрафик сохранён: task03_quality.png")
