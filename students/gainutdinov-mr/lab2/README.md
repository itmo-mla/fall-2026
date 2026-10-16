# Лабораторная работа №2. Метрическая классификация

В рамках лабораторной работы предстоит реализовать алгоритм классификации KNN и подобрать параметр k методом скользящего контроля.

На лекции были рассмотрены следующие алгоритмы:
* алгоритм метрической классификации KNN;
* метод окна Парзена;
* алгоритм отбора эталонов.

## Задание

1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?tags=13302-Classification);
2. реализовать алгоритм KNN с методом окна Парзена переменной ширины;
   1. в качестве ядра можно использовать гауссово ядро;
3. подобрать параметр k методом скользящего контроля (LOO);
4. обосновать выбор параметров алгоритма, построить графики эмпирического риска для различных k;
5. сравнить с [эталонной](https://scikit-learn.org/stable/) реализацией KNN;
   1. сравнить качество работы алгоритмов;
6. реализовать алгоритм отбора эталонов;
7. подготовить визуализацию результатов работы алгоритма отбора эталонов;
8. сравнить качество работы KNN с и без отбора эталонов; 
9. подготовить небольшой отчет о проделанной работе.
___
# Отчет
## Датасет
В качестве датасета был выбран Stellar Classification Dataset - SDSS17 с 3 классами: galaxy, star, quasar
![alt text](/students/gainutdinov-mr/lab2/source/report-images/image.png)

## Алгоритм KNN с методом окна Парзена переменной ширины
В классе ParzenClassifier реализован следующий алгоритм:
![alt text](/students/gainutdinov-mr/lab2/source/report-images/image-1.png)
В качесвте ядра было взято гауссовское - ParzenClassifier.gaussian_kernel, но класс поддерживает и другие.

## Подбор k методом скользящего контроля LOO
В main.py реализован методом select_k_loo(), однако упрощенная его запись может быть такова:
![alt text](/students/gainutdinov-mr/lab2/source/report-images/image-2.png)
Где algorithm - наш алгоритм классификации, которое зависит от k, в частности окна Парзена.
## Эталонная реализация KNN
![alt text](/students/gainutdinov-mr/lab2/source/report-images/image-3.png)

## Обоснуем k для метода скользящего контроля
![alt text](/students/gainutdinov-mr/lab2/source/figures/loo_risk.png)
При k = 1 достигается минимальная доля ошибок. Поэтому получаем 1NN

## Сравним качество работы алгоритмов

![alt text](/students/gainutdinov-mr/lab2/source/figures/sklearn_comparison.png)
Как мы видим, качеством мы не сильно отличаемся, зато по времени работы - значительно. Особенно это видно, когда мы работаем с большой выборкой, например 70 000. Предположительно много времени занимает перерасчет расстояний и их сортировка

## Реализуем алгоритмы отбора эталонов

Алгоритмы реализованы согласно слайдам в классе PrototypeSelector:
![alt text](/students/gainutdinov-mr/lab2/source/report-images/image-4.png)
Однако также были введены дополнительные ограничения:

Так, например, при жадном удалении, нам нельзя удалить из эталонов последнего представителя класса, иначе мы в дальнейшем не сможем этот класс различать.

Также есть ограничение, что мощность множества эталонов должна быть больше размера контрольной выборки k, ведь иначе мы сможем получить такое разбиение, где в обучающей выборке не будет ни одного эталона.

Для L = 250:

Получаем
![alt text](/students/gainutdinov-mr/lab2/source/figures/removal_history.png)
58 эталонов при жадном удалении

![alt text](/students/gainutdinov-mr/lab2/source/figures/addition_history.png)
15 эталонов при жадном добавлении


## Визуализируем результаты работы алгоритма отбора эталонов
Будем смотреть в проекциях:
### alpha + redhsift
Удаление
![alt text](/students/gainutdinov-mr/lab2/source/figures/removal_alpha_redshift.png)
Добавление
![alt text](/students/gainutdinov-mr/lab2/source/figures/addition_alpha_redshift.png)
### delta + redhsift
Удаление
![alt text](/students/gainutdinov-mr/lab2/source/figures/removal_delta_redshift.png)
Добавление
![alt text](/students/gainutdinov-mr/lab2/source/figures/addition_delta_redshift.png)

### u + redhsift
Удаление
![alt text](/students/gainutdinov-mr/lab2/source/figures/removal_u_redshift.png)
Добавление
![alt text](/students/gainutdinov-mr/lab2/source/figures/addition_u_redshift.png)

## Сравним качество работы KNN и отбора эталонов

![alt text](/students/gainutdinov-mr/lab2/source/figures/parzen_comparison.png)
При L = 2500 с более оптимизированным алгоритмом отбора получаем:
![alt text](/students/gainutdinov-mr/lab2/source/report-images/image-5.png)

## Вывод
В ходе работы был реализован KNN с окном Парзена переменной ширины и гауссовым ядром, а методом LOO выбран оптимальный параметр k = 1. На датасете SDSS17 реализация показала качество, близкое к KNN из scikit-learn, но уступила ему по скорости. Отбор эталонов позволил сократить время классификации при небольшом снижении качества.Предпочтительным оказался метод жадного удаления, который лучше сохраняет качество исходной модели, чем жадное добавление.
