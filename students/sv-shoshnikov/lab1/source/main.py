import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


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
        n_features = X.shape[1]
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
        correlations = np.nan_to_num(correlations, 0)
        self.w = correlations / (np.linalg.norm(correlations) + 1e-8)
        self.b = np.mean(y)
    
    def init_weights_random(self, n_features):
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
    
    def compute_optimal_lr(self, X_batch, y_batch, grad_w, grad_b):
        n = X_batch.shape[0]
        predictions = X_batch @ self.w + self.b
        margins = y_batch * predictions
        errors = -2 * (1 - margins) * y_batch
        
        numerator = np.sum(errors ** 2)
        denominator = grad_w @ (X_batch.T @ X_batch) @ grad_w + n * grad_b ** 2
        
        if denominator > 1e-8:
            optimal_lr = numerator / (denominator + 1e-8)
            return np.clip(optimal_lr, 0.001, 0.1)
        return self.lr
    
    def sample_by_margin(self, X, y, batch_size):
        margins = np.abs(self.compute_margin(X, y))
        probabilities = 1.0 / (margins + 1e-8)
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(X), size=min(batch_size, len(X)), 
                                   replace=False, p=probabilities)
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
                
                grad_w, grad_b = self.compute_gradient(X_batch, y_batch)
                
                if self.use_fastest_descent:
                    current_lr = self.compute_optimal_lr(X_batch, y_batch, grad_w, grad_b)
                else:
                    current_lr = self.lr
                
                self.v_w = self.momentum * self.v_w - current_lr * grad_w
                self.v_b = self.momentum * self.v_b - current_lr * grad_b
                
                self.w += self.v_w
                self.b += self.v_b
            
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")
    
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


def main():
    print("="*60)
    print("Лабораторная работа №1: Линейная классификация")
    print("="*60)
    
    print("\n1. Загрузка датасета Breast Cancer...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"   Train: {X_train.shape[0]} объектов, {X_train.shape[1]} признаков")
    print(f"   Test: {X_test.shape[0]} объектов")
    
    print("\n2. Обучение с инициализацией через корреляцию...")
    clf1 = LinearClassifier(learning_rate=0.01, momentum=0.9, l2_reg=0.001, n_epochs=100)
    clf1.fit(X_train, y_train, init_method='correlation')
    train_acc1 = clf1.score(X_train, y_train)
    test_acc1 = clf1.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc1:.4f}")
    print(f"   Test accuracy: {test_acc1:.4f}")
    visualize_margins(clf1, X_test, y_test, "Инициализация через корреляцию")
    
    print("\n3. Обучение со случайной инициализацией (мультистарт)...")
    best_clf = None
    best_acc = 0
    n_starts = 5
    for i in range(n_starts):
        np.random.seed(42 + i)
        clf = LinearClassifier(learning_rate=0.01, momentum=0.9, l2_reg=0.001, n_epochs=100)
        clf.fit(X_train, y_train, init_method='random')
        acc = clf.score(X_test, y_test)
        print(f"   Старт {i+1}: Test accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
    print(f"   Лучший результат: {best_acc:.4f}")
    visualize_margins(best_clf, X_test, y_test, "Мультистарт - лучшая модель")
    
    print("\n4. Обучение со скорейшим градиентным спуском...")
    clf_fastest = LinearClassifier(learning_rate=0.01, momentum=0.9, l2_reg=0.001, 
                                   n_epochs=100, use_fastest_descent=True)
    clf_fastest.fit(X_train, y_train, init_method='correlation')
    train_acc_fastest = clf_fastest.score(X_train, y_train)
    test_acc_fastest = clf_fastest.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_fastest:.4f}")
    print(f"   Test accuracy: {test_acc_fastest:.4f}")
    visualize_margins(clf_fastest, X_test, y_test, "Скорейший градиентный спуск")
    
    print("\n5. Обучение с предъявлением по модулю отступа...")
    clf_margin = LinearClassifier(learning_rate=0.01, momentum=0.9, l2_reg=0.001, 
                                  n_epochs=100, use_margin_sampling=True)
    clf_margin.fit(X_train, y_train, init_method='random')
    train_acc_margin = clf_margin.score(X_train, y_train)
    test_acc_margin = clf_margin.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_margin:.4f}")
    print(f"   Test accuracy: {test_acc_margin:.4f}")
    visualize_margins(clf_margin, X_test, y_test, "Предъявление по модулю отступа")
    
    print("\n6. Сравнение с эталонной реализацией (sklearn)...")
    baseline = SGDClassifier(
        loss='squared_hinge', penalty='l2', alpha=0.001,
        learning_rate='constant', eta0=0.01, max_iter=100, random_state=42
    )
    baseline.fit(X_train, y_train)
    baseline_train = baseline.score(X_train, y_train)
    baseline_test = baseline.score(X_test, y_test)
    print(f"   Train accuracy: {baseline_train:.4f}")
    print(f"   Test accuracy: {baseline_test:.4f}")
    
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
