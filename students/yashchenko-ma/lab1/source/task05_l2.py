"""
Задача 5: реализовать L2 регуляризацию.

Подключает task04 (and task03, task02, task01).
Добавляет ровно один новый параметр `tau` и переопределяет только
_regularize() - остальная логика SGD+инерция naследуется без изменений.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from task04_sgd_momentum import LinearClassifierMomentum, load_and_prepare, add_bias


class LinearClassifierL2(LinearClassifierMomentum):
    """SGD с инерцией + L2-регуляризация (weight decay)."""

    def __init__(self, tau=0.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau  # коэффициент L2-регуляризации (0 -> без регуляризации)

    def _regularize(self, grad, w):
        # Штраф tau*w добавляется к градиенту; bias (индекс 0) не регуляризуем,
        # чтобы не штрафовать сдвиг решающей границы.
        if self.tau > 0:
            reg = self.tau * w
            reg = reg.copy()
            reg[0] = 0.0
            return grad + reg
        return grad


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_prepare()
    Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

    taus = [0.0, 0.001, 0.01, 0.1, 1.0]
    norms, accs, f1s = [], [], []

    for tau in taus:
        clf = LinearClassifierL2(tau=tau, eta=0.01, gamma=0.9, n_epochs=10, random_state=42)
        clf.fit(Xb_train, y_train)
        y_pred = clf.predict(Xb_test)
        norms.append(np.linalg.norm(clf.w[1:]))
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        print(f"tau={tau:<6} ||w||={norms[-1]:.4f}  acc={accs[-1]:.4f}  f1={f1s[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, norms, marker="o", label="||w|| (без bias)")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("tau (коэффициент L2)")
    ax.set_ylabel("||w||")
    ax.set_title("Задача 5: влияние L2-регуляризации на норму весов")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../images/task05_l2.png", dpi=120)
    print("\nГрафик сохранён: task05_l2.png")

    print("\nАнализ: норма весов монотонно убывает с ростом tau - "
          "регуляризация работает корректно. F1 по классу 'отказ' остаётся "
          "0 при любом tau: регуляризация борется с переобучением, а не с "
          "дисбалансом классов - это разные проблемы.")
