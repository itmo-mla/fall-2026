import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .logistic_regression import LogisticRegressionIRLS


def main():
    # 1. Загрузка датасета
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 2. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Стандартизация признаков (критично для сходимости метода Ньютона)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Параметры модели
    C_value = 1.0  # Коэффициент регуляризации (обратный lambda)
    
    # 5. Обучение собственной модели (IRLS)
    print("Обучение собственной модели (Custom IRLS)...")
    custom_model = LogisticRegressionIRLS(max_iter=100, tol=1e-7, C=C_value)
    custom_model.fit(X_train_scaled, y_train)
    print(f"Количество итераций до сходимости: {custom_model.n_iter_}")
    
    # 6. Обучение эталонной модели (Scikit-Learn)
    print("\nОбучение эталонной модели (Scikit-Learn, solver='newton-cg')...")
    sklearn_model = LogisticRegression(
        l1_ratio = 0, 
        C=C_value, 
        solver='newton-cg', 
        max_iter=100, 
        tol=1e-7, 
        fit_intercept=True
    )
    sklearn_model.fit(X_train_scaled, y_train)
    print(f"Количество итераций: {sklearn_model.n_iter_}")
    
    # 7. Предсказания
    y_pred_custom = custom_model.predict(X_test_scaled)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    
    y_proba_custom = custom_model.predict_proba(X_test_scaled)[:, 1]
    y_proba_sklearn = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    
    # 8. Вычисление метрик
    acc_custom = accuracy_score(y_test, y_pred_custom)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    ll_custom = log_loss(y_test, y_proba_custom)
    ll_sklearn = log_loss(y_test, y_proba_sklearn)
    
    # 9. Сравнение весов
    weights_diff = np.linalg.norm(custom_model.coef_ - sklearn_model.coef_.flatten())
    intercept_diff = np.abs(custom_model.intercept_ - sklearn_model.intercept_[0])
    
    # 10. Вывод результатов
    print("Результаты")
    print(f"{'Метрика':<30} | {'Custom IRLS':<15} | {'Scikit-Learn':<15}")
    print(f"{'Accuracy (test)':<30} | {acc_custom:<15.4f} | {acc_sklearn:<15.4f}")
    print(f"{'Log-Loss (test)':<30} | {ll_custom:<15.4f} | {ll_sklearn:<15.4f}")
    print(f"{'Количество итераций':<30} | {custom_model.n_iter_} | {sklearn_model.n_iter_}")
    print(f"\nРазница норм весов (||w_custom - w_sklearn||): {weights_diff:.2e}")
    print(f"Разница свободных членов (|b_custom - b_sklearn|): {intercept_diff:.2e}")


if __name__ == "__main__":
    main()