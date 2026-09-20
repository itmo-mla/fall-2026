"""
Задача 7: реализовать предъявление объектов по модулю отступа.

Подключает task06 (and task05, task04, task03, task02, task01).
Переопределяет только _get_order().

LinearClassifier - финальный, полнофункциональный класс: SGD с инерцией
(задача 4) + L2 (задача 5) + скорейший спуск (задача 6) + margin-based
предъявление (задача 7). Именно его используют задачи 8-10.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score

from task06_steepest import LinearClassifierSteepest, load_and_prepare, add_bias


class LinearClassifier(LinearClassifierSteepest):
    """SGD с инерцией + L2 + скорейший спуск + margin-based предъявление."""

    def __init__(self, presentation="uniform", **kwargs):
        super().__init__(**kwargs)
        self.presentation = presentation  # 'uniform' | 'margin'

    def _get_order(self, rng, n, X, y):
        if self.presentation == "margin":
            # Пересчитываем отступы по текущим весам в начале эпохи и строим
            # распределение p_i ~ 1/(|M_i|+eps): объекты с малым по модулю
            # (или отрицательным) отступом предъявляются чаще.
            M_all = y * (X @ self.w)
            inv = 1.0 / (np.abs(M_all) + 0.1)
            probs = inv / inv.sum()
            return rng.choice(n, size=n, replace=True, p=probs)
        return rng.permutation(n)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare()
    Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

    common = dict(eta=0.01, gamma=0.9, n_epochs=10, step_mode="fixed", random_state=42)

    clf_uniform = LinearClassifier(presentation="uniform", **common)
    clf_uniform.fit(Xb_train, y_train)

    clf_margin = LinearClassifier(presentation="margin", **common)
    clf_margin.fit(Xb_train, y_train)

    for name, clf in [("Uniform", clf_uniform), ("Margin-based", clf_margin)]:
        y_pred = clf.predict(Xb_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        print(f"{name:15s} acc={acc:.4f}  recall(failure)={rec:.4f}  f1(failure)={f1:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(clf_uniform.Q_history, label="Uniform", linewidth=0.7)
    ax.plot(clf_margin.Q_history, label="Margin-based", linewidth=0.7)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Q")
    ax.set_title("Задача 7: равновероятное vs margin-based предъявление")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../images/task07_margin_presentation.png", dpi=120)
    print("\nГрафик сохранён: task07_margin_presentation.png")

    print("\nАнализ: margin-based предъявление частично решает проблему "
          "коллапса - чаще показываются 'сложные' объекты (малый/отрицательный "
          "отступ), а на старте обучения это преимущественно объекты редкого "
          "класса 'отказ'.")
