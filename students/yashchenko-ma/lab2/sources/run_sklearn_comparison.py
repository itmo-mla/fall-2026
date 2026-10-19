"""
Запускной скрипт для пункта 5 задания: сравнение своей реализации
KNN+Парзен с эталонной реализацией sklearn. Запускать из папки sources/:

    python run_sklearn_comparison.py

Только здесь используется готовая реализация из библиотеки
(sklearn.neighbors.KNeighborsClassifier) для сравнения "с эталонной реализацией").
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import load_and_split
from knn_parzen import ParzenKNN

'''
Окно Парзена vs KNN — в двух словах, и связь с профилем компактности

В двух словах: 
KNN — голосуют ровно k ближайших, все с весом 1 (жёсткая отсечка). 
Парзен — голосуют все, но чем дальше — тем тише голос (мягкое, плавное затухание).

Связь с профилем компактности П(m) (лекция, слайд 21: доля объектов, у которых m-й сосед — другого класса): 
        это по сути "профиль риска" по позиции соседа m=1,2,3,.... 
        И KNN, и Парзен — это просто два разных способа свернуть этот профиль в одно число:

KNN обрезает профиль жёстко на позиции k (все m≤k учитываются с весом 1, всё что дальше — с весом 0);
Парзен вместо обрезки берёт весь профиль целиком, но взвешивает каждую позицию m по функции ядра (чем дальше позиция — тем меньше вес).
'''

def main():
    X_train, X_test, y_train, y_test = load_and_split()
    k_opt = 1  # выбрано методом LOO, см. run_loo_selection.py

    own = ParzenKNN(k=k_opt).fit(X_train, y_train)
    pred_own = own.predict(X_test)

    skl_uniform = KNeighborsClassifier(n_neighbors=k_opt, weights="uniform").fit(X_train, y_train)
    pred_skl_uniform = skl_uniform.predict(X_test)

    skl_dist = KNeighborsClassifier(n_neighbors=k_opt, weights="distance").fit(X_train, y_train)
    pred_skl_dist = skl_dist.predict(X_test)

    print(f"k = {k_opt}\n")
    for name, pred in [
        ("Своя реализация (Парзен, Gaussian, переменная ширина, сумма по X^l)", pred_own),
        ("sklearn KNN (uniform, только k ближайших)", pred_skl_uniform),
        ("sklearn KNN (weights=distance, только k ближайших)", pred_skl_dist),
    ]:
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, pos_label=1)
        print(f"{name}:\n  accuracy={acc:.4f}  F1(class=1)={f1:.4f}")

    print("\nОтчет по своей реализации:")
    print(classification_report(y_test, pred_own, digits=3))

    print("--- сравнение по диапазону k (accuracy / F1 на тесте) ---")
    print(f"{'k':>3} | {'own_acc':>8} {'own_f1':>7} | {'skl_unif_acc':>13} {'skl_unif_f1':>12} | {'skl_dist_acc':>13} {'skl_dist_f1':>12}")
    for k in [1, 3, 5, 9, 15, 25, 40]:
        p_own = ParzenKNN(k=k).fit(X_train, y_train).predict(X_test)
        p_su = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_train, y_train).predict(X_test)
        p_sd = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_train, y_train).predict(X_test)
        row = (k,
               accuracy_score(y_test, p_own), f1_score(y_test, p_own, pos_label=1),
               accuracy_score(y_test, p_su), f1_score(y_test, p_su, pos_label=1),
               accuracy_score(y_test, p_sd), f1_score(y_test, p_sd, pos_label=1))
        print(f"{row[0]:>3} | {row[1]:>8.4f} {row[2]:>7.4f} | {row[3]:>13.4f} {row[4]:>12.4f} | {row[5]:>13.4f} {row[6]:>12.4f}")


if __name__ == "__main__":
    main()
