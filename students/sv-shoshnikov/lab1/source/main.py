import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, SGDClassifier


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def out_path(name):
    return os.path.join(OUT_DIR, name)


class LinearClassifier:
    STEP_RULES = ('constant', 'steepest', 'steepest_momentum')

    def __init__(self, learning_rate=0.01, momentum=0.9, l2_reg=0.01, n_epochs=100,
                 step_rule='constant', use_margin_sampling=False,
                 lr_min=1e-3, lr_max=1.0, clip_step=True, drop_last=False):
        """step_rule:
        'constant'          - SGD с инерцией, шаг learning_rate:  v = g*v + (1-g)*h*grad,  w -= v;
        'steepest'          - скорейший спуск без инерции: w -= h* * grad, h* минимизирует риск батча вдоль -grad;
        'steepest_momentum' - скорейший спуск с инерцией: направление d = g*d_prev + (1-g)*grad (скользящее
                              среднее градиентов), w -= h* * d, h* минимизирует риск батча вдоль фактического
                              направления d. Если <grad, d> <= 0, инерция сбрасывается (d = grad).
        Для 'steepest' параметр momentum не используется.
        clip_step ограничивает h* отрезком [lr_min, lr_max].
        drop_last отбрасывает неполный последний батч эпохи (455 mod 32 = 7 объектов): точный шаг по такому
        батчу слишком шумный."""
        if step_rule not in self.STEP_RULES:
            raise ValueError(f"step_rule должен быть одним из {self.STEP_RULES}")
        self.lr = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.n_epochs = n_epochs
        self.step_rule = step_rule
        self.use_margin_sampling = use_margin_sampling
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.clip_step = clip_step
        self.drop_last = drop_last
        self.n_steps_total = 0
        self.n_steps_clipped = 0
        self.step_history = []
        self.w = None
        self.b = None
        self.v_w = None
        self.v_b = None
        self.loss_history = []
        self.q_history = []
    
    def compute_margin(self, X, y):
        predictions = X @ self.w + self.b
        return y * predictions
    
    def compute_loss(self, X, y):
        margins = self.compute_margin(X, y)
        loss = np.mean((1 - margins) ** 2)
        l2_penalty = self.l2_reg * np.sum(self.w ** 2)
        return loss + l2_penalty
    
    def compute_gradient(self, X, y):
        n = X.shape[0]
        predictions = X @ self.w + self.b
        margins = y * predictions
        errors = -2 * (1 - margins) * y
        grad_w = (X.T @ errors) / n + 2 * self.l2_reg * self.w
        grad_b = np.mean(errors)
        return grad_w, grad_b
    
    def init_weights_correlation(self, X, y):
        self.w = (y @ X) / (X ** 2).sum(axis=0)
        self.b = 0.0
    
    def init_weights_random(self, n_features):
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
    
    def compute_optimal_lr(self, X_batch, grad_w, grad_b, dir_w, dir_b):
        """Шаг h*, минимизирующий риск батча вдоль направления d = (dir_w, dir_b):
        Q(h) = mean((1 - M_i + h*y_i*s_i)^2) + l2*||w - h*dir_w||^2,  s_i = <dir_w, x_i> + dir_b.
        Q - парабола по h, поэтому h* = <grad, d> / (2*(mean(s^2) + l2*||dir_w||^2)).
        Для d = grad это ||grad||^2 / (2*(mean(s^2) + l2*||grad_w||^2))."""
        s = X_batch @ dir_w + dir_b
        numerator = grad_w @ dir_w + grad_b * dir_b
        denominator = 2 * (np.mean(s ** 2) + self.l2_reg * (dir_w @ dir_w))
        if denominator <= 1e-12:
            return self.lr
        h = numerator / denominator
        self.n_steps_total += 1
        if self.clip_step:
            clipped = float(np.clip(h, self.lr_min, self.lr_max))
            self.n_steps_clipped += int(clipped != h)
            h = clipped
        self.step_history.append(h)
        return h

    def sample_by_margin(self, X, y, batch_size, temperature=1.0):
        margins = np.abs(self.compute_margin(X, y))
        probabilities = np.exp(-(margins - margins.min()) / temperature)
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(X), size=min(batch_size, len(X)), 
                                   replace=True, p=probabilities)
        return X[indices], y[indices]
    
    def fit(self, X, y, init_method='correlation'):
        n_samples, n_features = X.shape
        
        if init_method == 'correlation':
            self.init_weights_correlation(X, y)
        else:
            self.init_weights_random(n_features)
        
        self.v_w = np.zeros(n_features)
        self.v_b = 0.0
        self.d_w = np.zeros(n_features)
        self.d_b = 0.0
        self.loss_history = []
        self.n_steps_total = 0
        self.n_steps_clipped = 0
        self.step_history = []
        Q = self.compute_loss(X, y)
        self.q_history = [Q]
        self.loss_history = [Q]
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            batch_size = 32
            stop = n_samples - n_samples % batch_size if self.drop_last else n_samples
            for i in range(0, stop, batch_size):
                if self.use_margin_sampling:
                    X_batch, y_batch = self.sample_by_margin(X, y, batch_size)
                else:
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]
                
                batch_loss = self.compute_loss(X_batch, y_batch)
                grad_w, grad_b = self.compute_gradient(X_batch, y_batch)
                
                if self.step_rule == 'constant':
                    self.v_w = self.momentum * self.v_w + (1 - self.momentum) * self.lr * grad_w
                    self.v_b = self.momentum * self.v_b + (1 - self.momentum) * self.lr * grad_b
                else:
                    if self.step_rule == 'steepest_momentum':
                        dir_w = self.momentum * self.d_w + (1 - self.momentum) * grad_w
                        dir_b = self.momentum * self.d_b + (1 - self.momentum) * grad_b
                        if grad_w @ dir_w + grad_b * dir_b <= 0:
                            dir_w, dir_b = grad_w, grad_b
                    else:
                        dir_w, dir_b = grad_w, grad_b
                    self.d_w, self.d_b = dir_w, dir_b
                    h = self.compute_optimal_lr(X_batch, grad_w, grad_b, dir_w, dir_b)
                    self.v_w = h * dir_w
                    self.v_b = h * dir_b
                
                self.w -= self.v_w
                self.b -= self.v_b
                
                lam = len(y_batch) / n_samples
                Q = lam * batch_loss + (1 - lam) * Q
            
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            self.q_history.append(Q)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}, Q (рекуррентная): {Q:.4f}")
    
    def predict(self, X):
        return np.sign(X @ self.w + self.b)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    y = 2 * y - 1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def visualize_margins(clf, X, y, title):
    margins = clf.compute_margin(X, y)
    
    plt.figure(figsize=(10, 6))
    plt.hist(margins, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Граница решения')
    plt.xlabel('Отступ', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(f'{title.replace(" ", "_")}.png'), dpi=300)
    plt.close()
    
    print(f"\n{title}:")
    print(f"  Средний отступ: {np.mean(margins):.4f}")
    print(f"  Мин отступ: {np.min(margins):.4f}")
    print(f"  Макс отступ: {np.max(margins):.4f}")
    print(f"  Ошибки (отступ < 0): {np.sum(margins < 0)}/{len(margins)}")


def visualize_loss(loss_histories, labels):
    plt.figure(figsize=(10, 6))
    for loss_history, label in zip(loss_histories, labels):
        plt.plot(loss_history, label=label, linewidth=2)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Функция потерь', fontsize=12)
    plt.title('История обучения', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path('loss_history.png'), dpi=300)
    plt.close()


def classification_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))
    tn = np.sum((y_pred == -1) & (y_true == -1))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1, np.array([[tn, fp], [fn, tp]])


def print_metrics(name, y_true, y_pred):
    precision, recall, f1, cm = classification_metrics(y_true, y_pred)
    print(f"   {name}: accuracy {np.mean(y_true == y_pred):.4f}, precision {precision:.4f}, "
          f"recall {recall:.4f}, f1 {f1:.4f}")
    print(f"   Матрица ошибок (строки - истинный класс, столбцы - предсказанный): {cm.tolist()}")


def check_line_search(clf, X, y, batch_size=32, seed=0):
    """Проверка скорейшего шага: на батче риск вдоль направления d должен быть минимален при h = h*."""
    idx = np.random.RandomState(seed).choice(len(X), batch_size, replace=False)
    Xb, yb = X[idx], y[idx]
    gw, gb = clf.compute_gradient(Xb, yb)
    clip, clf.clip_step = clf.clip_step, False
    h = clf.compute_optimal_lr(Xb, gw, gb, gw, gb)
    clf.clip_step = clip
    w0, b0 = clf.w.copy(), clf.b

    def risk_at(step):
        clf.w, clf.b = w0 - step * gw, b0 - step * gb
        return clf.compute_loss(Xb, yb)

    grid = h * np.linspace(0.0, 3.0, 3001)
    values = np.array([risk_at(t) for t in grid])
    result = (h, grid[np.argmin(values)], risk_at(0.0), risk_at(h), risk_at(0.9 * h), risk_at(1.1 * h))
    clf.w, clf.b = w0, b0
    return result


def visualize_step_sizes(runs, filename='step_sizes.png'):
    plt.figure(figsize=(10, 6))
    for clf, label in runs:
        plt.plot(clf.step_history, label=label, linewidth=1)
    first = runs[0][0]
    plt.axhline(first.lr_max, color='r', linestyle='--', label='clip: верхняя граница')
    plt.axhline(first.lr_min, color='g', linestyle='--', label='clip: нижняя граница')
    plt.yscale('log')
    plt.xlabel('Номер шага (батча)', fontsize=12)
    plt.ylabel('Шаг h*', fontsize=12)
    plt.title('Шаг скорейшего спуска', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(filename), dpi=300)
    plt.close()


def check_gradient(clf, X, y, eps=1e-6):
    grad_w, grad_b = clf.compute_gradient(X, y)
    analytic = np.append(grad_w, grad_b)
    params = np.append(clf.w, clf.b)
    numeric = np.zeros_like(params)
    for j in range(len(params)):
        step = np.zeros_like(params)
        step[j] = eps
        clf.w, clf.b = (params + step)[:-1], (params + step)[-1]
        up = clf.compute_loss(X, y)
        clf.w, clf.b = (params - step)[:-1], (params - step)[-1]
        down = clf.compute_loss(X, y)
        numeric[j] = (up - down) / (2 * eps)
    clf.w, clf.b = params[:-1], params[-1]
    return np.abs(analytic - numeric).max()


def visualize_recurrent_q(clf, filename='recurrent_q.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(clf.loss_history, label='Риск по всей выборке', linewidth=2)
    plt.plot(clf.q_history, label='Рекуррентная оценка Q', linewidth=2)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Функционал качества', fontsize=12)
    plt.title('Рекуррентная оценка функционала качества', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path(filename), dpi=300)
    plt.close()


def main():
    np.random.seed(42)
    print("="*60)
    print("Лабораторная работа №1: Линейная классификация")
    print("="*60)
    
    print("\n1. Загрузка датасета Breast Cancer...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"   Train: {X_train.shape[0]} объектов, {X_train.shape[1]} признаков")
    print(f"   Test: {X_test.shape[0]} объектов")
    
    print("\n2. Обучение с инициализацией через корреляцию...")
    clf1 = LinearClassifier(learning_rate=0.1, momentum=0.9, l2_reg=0.001, n_epochs=100)
    clf1.fit(X_train, y_train, init_method='correlation')
    train_acc1 = clf1.score(X_train, y_train)
    test_acc1 = clf1.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc1:.4f}")
    print(f"   Test accuracy: {test_acc1:.4f}")
    visualize_margins(clf1, X_test, y_test, "Инициализация через корреляцию")
    visualize_recurrent_q(clf1)
    print(f"   Q (рекуррентная) = {clf1.q_history[-1]:.4f}, риск по выборке = {clf1.loss_history[-1]:.4f}")
    print_metrics("Тест", y_test, clf1.predict(X_test))
    print(f"   Макс. расхождение аналитич. и численного градиента: {check_gradient(clf1, X_train, y_train):.2e}")
    
    print("\n3. Обучение со случайной инициализацией (мультистарт)...")
    best_clf = None
    best_risk = np.inf
    n_starts = 5
    for i in range(n_starts):
        np.random.seed(42 + i)
        clf = LinearClassifier(learning_rate=0.1, momentum=0.9, l2_reg=0.001, n_epochs=100)
        clf.fit(X_train, y_train, init_method='random')
        risk = clf.compute_loss(X_train, y_train)
        print(f"   Старт {i+1}: риск на обучении = {risk:.4f}")
        if risk < best_risk:
            best_risk = risk
            best_clf = clf
    best_acc = best_clf.score(X_test, y_test)
    print(f"   Лучший запуск (по риску на обучении {best_risk:.4f}): Test accuracy = {best_acc:.4f}")
    visualize_margins(best_clf, X_test, y_test, "Мультистарт - лучшая модель")
    
    print("\n4. Обучение со скорейшим градиентным спуском...")
    clf_fastest = LinearClassifier(l2_reg=0.001, n_epochs=100, step_rule='steepest', drop_last=True)
    clf_fastest.fit(X_train, y_train, init_method='correlation')
    train_acc_fastest = clf_fastest.score(X_train, y_train)
    test_acc_fastest = clf_fastest.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_fastest:.4f}")
    print(f"   Test accuracy: {test_acc_fastest:.4f}")
    print(f"   Шагов: {clf_fastest.n_steps_total}, из них с ограничением clip: {clf_fastest.n_steps_clipped}")
    visualize_margins(clf_fastest, X_test, y_test, "Скорейший градиентный спуск")
    
    print("\n5. Обучение с предъявлением по модулю отступа...")
    clf_margin = LinearClassifier(learning_rate=0.1, momentum=0.9, l2_reg=0.001, 
                                  n_epochs=100, use_margin_sampling=True)
    clf_margin.fit(X_train, y_train, init_method='random')
    train_acc_margin = clf_margin.score(X_train, y_train)
    test_acc_margin = clf_margin.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_margin:.4f}")
    print(f"   Test accuracy: {test_acc_margin:.4f}")
    visualize_margins(clf_margin, X_test, y_test, "Предъявление по модулю отступа")
    
    print("\n5b. Скорейший спуск: проверка шага, инерция и ограничение clip...")
    chk = LinearClassifier(l2_reg=0.001)
    chk.init_weights_correlation(X_train, y_train)
    h, h_grid, q0, q_h, q_lo, q_hi = check_line_search(chk, X_train, y_train)
    print(f"   Старт (корреляция): h* = {h:.5f}, минимум на сетке при h = {h_grid:.5f}")
    print(f"   Риск батча: до шага {q0:.4f}, при h* {q_h:.4f}, при 0.9h* {q_lo:.4f}, при 1.1h* {q_hi:.4f}")
    np.random.seed(42)
    chk.fit(X_train, y_train, init_method='correlation')
    h, h_grid, q0, q_h, q_lo, q_hi = check_line_search(chk, X_train, y_train)
    print(f"   Обученная модель: h* = {h:.5f}, минимум на сетке при h = {h_grid:.5f}, "
          f"риск {q_h:.5f} (0.9h*: {q_lo:.5f}, 1.1h*: {q_hi:.5f})")

    variants = {
        'Скорейший, без инерции (clip)': dict(step_rule='steepest', drop_last=True),
        'Скорейший, без инерции (без clip)': dict(step_rule='steepest', clip_step=False, drop_last=True),
        'Скорейший с инерцией (clip)': dict(step_rule='steepest_momentum', momentum=0.9, drop_last=True),
        'Скорейший с инерцией (без clip)': dict(step_rule='steepest_momentum', momentum=0.9, clip_step=False,
                                                drop_last=True),
        'SGD без инерции, h = 0.01': dict(step_rule='constant', learning_rate=0.01, momentum=0.0),
    }
    variant_runs = []
    for name, kw in variants.items():
        np.random.seed(42)
        c = LinearClassifier(l2_reg=0.001, n_epochs=100, **kw)
        c.fit(X_train, y_train, init_method='correlation')
        variant_runs.append((c, name))
        share = f"{c.n_steps_clipped}/{c.n_steps_total}" if c.step_rule != 'constant' else "-"
        hs = c.step_history
        h_info = f", h: min {min(hs):.4f}, медиана {np.median(hs):.4f}, max {max(hs):.4f}" if hs else ""
        print(f"   {name}: риск {c.loss_history[-1]:.4f}, train {c.score(X_train, y_train):.4f}, "
              f"test {c.score(X_test, y_test):.4f}, шагов с clip {share}{h_info}")
    visualize_step_sizes([r for r in variant_runs if r[0].step_rule != 'constant'])

    print("\n5c. Устойчивость к случайности: 10 запусков (seed 0..9), инициализация через корреляцию...")
    configs = {
        'SGD с инерцией, h = 0.1, gamma = 0.9': dict(step_rule='constant', learning_rate=0.1, momentum=0.9),
        'SGD без инерции, h = 0.01': dict(step_rule='constant', learning_rate=0.01, momentum=0.0),
        'Скорейший, без инерции, неполный батч': dict(step_rule='steepest'),
        'Скорейший, без инерции, без неполного батча': dict(step_rule='steepest', drop_last=True),
        'Скорейший с инерцией, без неполного батча': dict(step_rule='steepest_momentum', momentum=0.9,
                                                          drop_last=True),
    }
    print(f"   {'метод':<46} {'риск (среднее / макс)':>22} {'test (среднее / мин)':>22}")
    for name, kw in configs.items():
        rows = []
        for seed in range(10):
            np.random.seed(seed)
            c = LinearClassifier(l2_reg=0.001, n_epochs=100, **kw)
            c.fit(X_train, y_train, init_method='correlation')
            rows.append((c.loss_history[-1], c.score(X_test, y_test)))
        rows = np.array(rows)
        print(f"   {name:<46} {rows[:, 0].mean():>11.4f} /{rows[:, 0].max():>8.4f} "
              f"{rows[:, 1].mean():>11.4f} /{rows[:, 1].min():>8.4f}")

    print("\n6. Сравнение с эталонной реализацией (sklearn)...")
    l2 = 0.001
    baseline = SGDClassifier(
        loss='squared_error', penalty='l2', alpha=2 * l2,
        learning_rate='constant', eta0=0.01, max_iter=100, tol=None, random_state=42
    )
    baseline.fit(X_train, y_train)
    baseline_test = baseline.score(X_test, y_test)
    print_metrics("sklearn SGDClassifier", y_test, baseline.predict(X_test))
    ridge = RidgeClassifier(alpha=len(y_train) * l2)
    ridge.fit(X_train, y_train)
    print_metrics("sklearn RidgeClassifier", y_test, ridge.predict(X_test))
    print_metrics("Своя реализация", y_test, clf1.predict(X_test))
    
    print("\n7. Визуализация истории обучения...")
    visualize_loss([clf1.loss_history, best_clf.loss_history, clf_fastest.loss_history, clf_margin.loss_history,
                    variant_runs[2][0].loss_history], 
                   ['Корреляция', 'Мультистарт', 'Скорейший спуск (без инерции)', 'По модулю отступа',
                    'Скорейший спуск с инерцией'])
    
    print("\n" + "="*60)
    print("ИТОГИ:")
    print("="*60)
    print(f"Корреляция:              Test = {test_acc1:.4f}")
    print(f"Мультистарт:             Test = {best_acc:.4f}")
    print(f"Скорейший спуск:         Test = {test_acc_fastest:.4f}")
    print(f"По модулю отступа:       Test = {test_acc_margin:.4f}")
    print(f"Эталон (sklearn):        Test = {baseline_test:.4f}")
    print("\nВсе графики сохранены в директорию lab1.")
    print("="*60)


if __name__ == "__main__":
    main()
