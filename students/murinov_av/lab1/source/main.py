import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import FastestGradientDescentClassifier, LinearSGDClassifier


def load_data(test_size=0.3, random_state=42):
    data = load_breast_cancer()

    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    y = np.where(y == 0, -1.0, 1.0)

    X_train_raw, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, scores):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, scores)),
        'f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'n_errors': int(np.sum(y_true != y_pred)),
    }


def evaluate_model(model, X, y, name=''):
    scores = model.decision_function(X).ravel()
    y_pred = model.predict(X)

    metrics = compute_metrics(y, y_pred, scores)
    margins = scores * y

    if name:
        print(f'\n{name}:')
        print(f"  accuracy  = {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC   = {metrics['roc_auc']:.4f}")
        print(f"  F1-score  = {metrics['f1']:.4f}")
        print(f"  precision = {metrics['precision']:.4f}")
        print(f"  recall    = {metrics['recall']:.4f}")
        print(f"  ошибок    = {metrics['n_errors']}/{len(y)}")

    return metrics, margins


def plot_margins(margins, title, save_path):
    sorted_idx = np.argsort(margins)
    sorted_margins = margins[sorted_idx]

    colors = np.where(
        sorted_margins < 0,
        'red',
        np.where(sorted_margins < 1, 'gold', 'limegreen')
    )

    plt.figure(figsize=(12, 5))
    plt.bar(
        range(len(sorted_margins)),
        sorted_margins,
        color=colors,
        width=1.0
    )

    plt.axhline(0, color='black', linewidth=1)
    plt.axhline(1, color='blue', linewidth=1, linestyle='--')

    plt.xlabel('Объекты (ранжированы по возрастанию отступа)')
    plt.ylabel('Отступ $M_i(w)$')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_quality(history, label, title, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(history, lw=0.8, label=label, color='steelblue')

    plt.xlabel('Итерация')
    plt.ylabel(r'$\tilde{Q}$')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()


def add_result(results, key, metrics, **extra):
    record = dict(metrics)

    for field, value in extra.items():
        if isinstance(value, (np.floating, float)):
            record[field] = float(value)
        elif isinstance(value, (np.integer, int)):
            record[field] = int(value)
        else:
            record[field] = value

    results[key] = record


def run_experiment(
    key,
    title,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    results,
    make_plots=True,
    **extra
):
    """Обучение, оценка, графики и сохранение результата."""
    model.fit(X_train, y_train)

    metrics, margins = evaluate_model(
        model,
        X_test,
        y_test,
        name=title
    )

    if make_plots:
        plot_margins(
            margins,
            title=f'Отступы: {title}',
            save_path=os.path.join('reports', f'{key}_margins.png')
        )

        if getattr(model, 'q_history_', None):
            plot_quality(
                model.q_history_,
                title,
                title=f'Кривая качества: {title}',
                save_path=os.path.join('reports', f'{key}_quality.png')
            )

    weight = getattr(model, 'w_', None)
    q_history = getattr(model, 'q_history_', None)

    add_result(
        results,
        key,
        metrics,
        weight_norm=float(np.linalg.norm(weight)) if weight is not None else None,
        q_final=float(q_history[-1]) if q_history else None,
        **extra
    )

    return model


def run_multistart(
    X_train,
    y_train,
    X_test,
    y_test,
    results,
    n_starts=7
):
    """Мультистарт с выбором лучшей модели по train accuracy."""
    best_model = None
    best_seed = None
    best_train_accuracy = -1.0

    print('\nМультистарт:')

    for s in range(n_starts):
        seed = 100 + s

        model = LinearSGDClassifier(
            lr=0.01,
            gamma=0.9,
            epochs=50,
            init='random',
            sampling='random',
            seed=seed
        ).fit(X_train, y_train)

        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        print(f'  старт #{s}: train accuracy={train_acc:.4f}')

        if train_acc > best_train_accuracy:
            best_train_accuracy = train_acc
            best_seed = seed
            best_model = model

    metrics, margins = evaluate_model(
        best_model,
        X_test,
        y_test,
        name='Мультистарт (лучший)'
    )

    plot_margins(
        margins,
        title='Отступы: мультистарт, лучшая модель',
        save_path=os.path.join('reports', 'multistart_best_margins.png')
    )

    if best_model.q_history_:
        plot_quality(
            best_model.q_history_,
            'multistart best',
            title='Кривая качества: мультистарт, лучшая модель',
            save_path=os.path.join('reports', 'multistart_best_quality.png')
        )

    add_result(
        results,
        'multistart_best',
        metrics,
        weight_norm=float(np.linalg.norm(best_model.w_)),
        q_final=float(best_model.q_history_[-1]) if best_model.q_history_ else None,
        best_seed=best_seed,
        best_train_accuracy=best_train_accuracy
    )


def run_reference_model(X_train, y_train, X_test, y_test, results):
    """Эталонная модель"""
    model = RidgeClassifier(alpha=1.0)

    model.fit(X_train, y_train)

    metrics, _ = evaluate_model(
            model,
            X_test,
            y_test,
            name='Эталон'
        )

    add_result(
            results,
            'reference',
            metrics
        )


def save_results(results):
    results_path = os.path.join('reports', 'results.json')

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\nРезультаты сохранены в {results_path}')


def main():
    os.makedirs('reports', exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    results = {}

    experiments = [
        (
            'sgd_random_init',
            'SGD+Momentum (random init)',
            LinearSGDClassifier(
                lr=0.01,
                gamma=0.9,
                epochs=100,
                init='random',
                sampling='random',
                seed=42
            )
        ),
        (
            'sgd_correlation_init',
            'SGD+Momentum (correlation init)',
            LinearSGDClassifier(
                lr=0.01,
                gamma=0.9,
                epochs=100,
                init='correlation',
                sampling='random',
                seed=42
            )
        ),
        (
            'sgd_l2',
            'SGD+Momentum + L2',
            LinearSGDClassifier(
                lr=0.01,
                gamma=0.9,
                epochs=100,
                init='correlation',
                sampling='random',
                reg_tau=1e-3,
                seed=42
            )
        ),
        (
            'fastest_gradient_descent',
            'Скорейший градиентный спуск',
            FastestGradientDescentClassifier(
                epochs=100,
                reg_tau=0.0,
                seed=42
            )
        ),
    ]

    for key, title, model in experiments:
        run_experiment(
            key,
            title,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            results
        )

    run_multistart(
        X_train,
        y_train,
        X_test,
        y_test,
        results,
        n_starts=7
    )

    run_experiment(
        'sgd_margin_sampling',
        'SGD+Momentum (sampling by |margin|)',
        LinearSGDClassifier(
            lr=0.01,
            gamma=0.9,
            epochs=100,
            init='correlation',
            sampling='margin',
            mu_plus=2.0,
            mu_minus=-1.0,
            seed=42
        ),
        X_train,
        y_train,
        X_test,
        y_test,
        results
    )

    run_reference_model(
        X_train,
        y_train,
        X_test,
        y_test,
        results
    )

    save_results(results)


if __name__ == '__main__':
    main()