# Лабораторная работа №2. Метрическая классификация

## Датасет

Breast Cancer Wisconsin из `sklearn.datasets`: 569 объектов, 30 числовых признаков и 2 класса — злокачественная и доброкачественная опухоль.

Метки перекодированы в формат $y \in \{-1, +1\}$. Подготовка данных: стратифицированное разбиение 80/20 (455 объектов на обучение, 114 на тест)

## Запуск

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd source
python main.py
```
