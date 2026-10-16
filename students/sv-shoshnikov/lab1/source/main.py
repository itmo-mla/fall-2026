import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, SGDClassifier


class LinearClassifier:
    def __init__(self, learning_rate=0.01, momentum=0.9, l2_reg=0.01, n_epochs=100, 
                 use_fastest_descent=False, use_margin_sampling=False):
        self.lr = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.n_epochs = n_epochs
        self.use_fastest_descent = use_fastest_descent
        self.use_margin_sampling = use_margin_sampling
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
    
    def compute_optimal_lr(self, X_batch, y_batch, grad_w, grad_b):
        direction = X_batch @ grad_w + grad_b
        
        numerator = grad_w @ grad_w + grad_b ** 2
        denominator = 2 * (np.mean(direction ** 2) + self.l2_reg * (grad_w @ grad_w))
        
        if denominator > 1e-8:
            optimal_lr = numerator / denominator
            return np.clip(optimal_lr, 0.001, 1.0)
        return self.lr
    
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
        self.loss_history = []
        Q = self.compute_loss(X, y)
        self.q_history = [Q]
        self.loss_history = [Q]
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            batch_size = 32
            for i in range(0, n_samples, batch_size):
                if self.use_margin_sampling:
                    X_batch, y_batch = self.sample_by_margin(X, y, batch_size)
                else:
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]
                
                batch_loss = self.compute_loss(X_batch, y_batch)
                grad_w, grad_b = self.compute_gradient(X_batch, y_batch)
                
                if self.use_fastest_descent:
                    current_lr = self.compute_optimal_lr(X_batch, y_batch, grad_w, grad_b)
                else:
                    current_lr = self.lr
                
                self.v_w = self.momentum * self.v_w + (1 - self.momentum) * current_lr * grad_w
                self.v_b = self.momentum * self.v_b + (1 - self.momentum) * current_lr * grad_b
                
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
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300)
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
    plt.savefig('loss_history.png', dpi=300)
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
    plt.savefig(filename, dpi=300)
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
    clf_fastest = LinearClassifier(learning_rate=0.1, momentum=0.9, l2_reg=0.001, 
                                   n_epochs=100, use_fastest_descent=True)
    clf_fastest.fit(X_train, y_train, init_method='correlation')
    train_acc_fastest = clf_fastest.score(X_train, y_train)
    test_acc_fastest = clf_fastest.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_fastest:.4f}")
    print(f"   Test accuracy: {test_acc_fastest:.4f}")
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
    visualize_loss([clf1.loss_history, best_clf.loss_history, clf_fastest.loss_history, clf_margin.loss_history], 
                   ['Корреляция', 'Мультистарт', 'Скорейший спуск', 'По модулю отступа'])
    
    print("\n" + "="*60)
    print("ИТОГИ:")
    print("="*60)
    print(f"Корреляция:              Test = {test_acc1:.4f}")
    print(f"Мультистарт:             Test = {best_acc:.4f}")
    print(f"Скорейший спуск:         Test = {test_acc_fastest:.4f}")
    print(f"По модулю отступа:       Test = {test_acc_margin:.4f}")
    print(f"Эталон (sklearn):        Test = {baseline_test:.4f}")
    print("\nВсе графики сохранены в текущей директории.")
    print("="*60)


if __name__ == "__main__":
    main()
