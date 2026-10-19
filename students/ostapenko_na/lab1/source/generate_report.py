from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

METRICS_PATH = RESULTS_DIR / "metrics.csv"
MULTISTART_PATH = RESULTS_DIR / "multistart_results.csv"
README_PATH = BASE_DIR / "README.md"

REFERENCE_NAME = "sklearn SGDClassifier"

TABLE_MODEL_NAMES = {
    "Correlation + random": "Corr. + random",
    "Random + multistart": "Multistart",
    "Correlation + |margin|": "Corr. + margin",
    "Steepest gradient": "Steepest",
    "sklearn SGDClassifier": "sklearn SGD",
}


def fmt(value, digits=4):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def check_inputs():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл {METRICS_PATH.name}. "
            "Сначала запусти linear_classifier.py."
        )

    missing_images = [
        name
        for name in [
            "q_history.png",
            "train_loss.png",
            "metrics_comparison.png",
            "roc_curves.png",
            "margins.png",
            "multistart.png",
        ]
        if not (IMAGES_DIR / name).exists()
    ]

    if missing_images:
        print(
            "Предупреждение: отсутствуют графики: "
            + ", ".join(missing_images)
        )


def build_results_table(metrics):
    columns = [
        "model",
        "train_loss",
        "train_accuracy",
        "test_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    table = metrics[columns].copy()
    table.columns = [
        "Модель",
        "Train loss",
        "Train acc.",
        "Test acc.",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
    ]


    table["Модель"] = table["Модель"].map(
        lambda name: TABLE_MODEL_NAMES.get(name, name)
    )

    for col in table.columns[1:]:
        table[col] = table[col].map(fmt)

    return markdown_table(table)


def build_multistart_text():
    if not MULTISTART_PATH.exists():
        return (
            "Выполнено несколько запусков со случайными начальными весами. "
            "Лучший запуск выбирался по минимальному среднему loss на train."
        )

    starts = pd.read_csv(MULTISTART_PATH)
    best = starts.loc[starts["train_loss"].idxmin()]

    return (
        f"Было выполнено {len(starts)} независимых запусков. "
        f"Лучшим оказался запуск №{int(best['start'])} с "
        f"train loss = {best['train_loss']:.6f}. "
        "Лучший запуск выбирался только по train-выборке; "
        "test-выборка при выборе не использовалась."
    )


def build_results_analysis(metrics):
    custom = metrics[metrics["model"] != REFERENCE_NAME].copy()
    reference = metrics[metrics["model"] == REFERENCE_NAME].copy()

    paragraphs = []

    best_acc_value = custom["test_accuracy"].max()
    best_acc = custom[custom["test_accuracy"] == best_acc_value].copy()
    best_acc_names = ", ".join(
        f"`{name}`" for name in best_acc["model"].tolist()
    )

    best_auc_value = custom["roc_auc"].max()
    best_auc = custom[custom["roc_auc"] == best_auc_value]
    best_auc_names = ", ".join(
        f"`{name}`" for name in best_auc["model"].tolist()
    )

    paragraphs.append(
        f"Максимальная test accuracy среди собственных реализаций составляет "
        f"**{best_acc_value:.4f}** и получена для: {best_acc_names}. "
        f"Максимальный ROC AUC среди собственных реализаций составляет "
        f"**{best_auc_value:.6f}** и получен для: {best_auc_names}."
    )

    random_row = custom[custom["model"] == "Correlation + random"]
    margin_row = custom[custom["model"] == "Correlation + |margin|"]

    if not random_row.empty and not margin_row.empty:
        rnd = random_row.iloc[0]
        mar = margin_row.iloc[0]
        paragraphs.append(
        "Для сравнения способов предъявления объектов были обучены две "
        "модели с одинаковой корреляционной инициализацией весов: "
        "со случайным предъявлением и с предъявлением по модулю отступа. "
        f"Test accuracy составила **{rnd['test_accuracy']:.4f}** и "
        f"**{mar['test_accuracy']:.4f}** соответственно, "
        f"F1 — **{rnd['f1']:.4f}** и **{mar['f1']:.4f}**, "
        f"ROC AUC — **{rnd['roc_auc']:.6f}** и "
        f"**{mar['roc_auc']:.6f}**. "
        "Таким образом, предъявление объектов по модулю отступа дало "
        "небольшое улучшение test accuracy и F1 по сравнению со случайным "
        "предъявлением. При этом ROC AUC незначительно снизился. "
        "Следовательно, способ предъявления объектов повлиял на метрики "
        "неоднозначно: качество классификации при фиксированном пороге "
        "несколько улучшилось, тогда как качество ранжирования практически "
        "не изменилось и немного снизилось."
    )

    steepest = custom[custom["model"] == "Steepest gradient"]
    if not steepest.empty:
        row = steepest.iloc[0]
        paragraphs.append(
            f"Скорейший градиентный спуск показал test accuracy "
            f"**{row['test_accuracy']:.4f}**, F1 = **{row['f1']:.4f}** и "
            f"ROC AUC = **{row['roc_auc']:.4f}**. Его train loss "
            f"(**{row['train_loss']:.4f}**) выше, чем у основных вариантов SGD. "
            "В данном стохастическом эксперименте выбор шага по отдельному "
            "объекту приводит к менее стабильной траектории обучения и худшему "
            "итоговому качеству."
        )

    if not reference.empty:
        ref = reference.iloc[0]
        best = best_acc.sort_values(
            ["roc_auc", "f1"], ascending=False
        ).iloc[0]

        paragraphs.append(
            "Для сравнения с эталонным алгоритмом выбрана собственная "
            f"реализация `{best['model']}`, поскольку она показала максимальную "
            "test accuracy среди собственных реализаций. "
            f"Её test accuracy = **{best['test_accuracy']:.4f}**, "
            f"F1 = **{best['f1']:.4f}**, ROC AUC = **{best['roc_auc']:.6f}**. "
            f"У эталонного `SGDClassifier`: test accuracy = "
            f"**{ref['test_accuracy']:.4f}**, F1 = **{ref['f1']:.4f}**, "
            f"ROC AUC = **{ref['roc_auc']:.6f}**. "
            "Полученные значения близки, поэтому собственная реализация SGD "
            "показывает качество, сопоставимое с библиотечным эталоном."
        )

        if ref["roc_auc"] >= 1.0 - 1e-12 and ref["test_accuracy"] < 1.0:
            paragraphs.append(
                "ROC AUC = **1.0** у эталонной модели не противоречит "
                "accuracy < 1. ROC AUC оценивает качество ранжирования объектов "
                "по непрерывному score, а accuracy — классификацию при конкретном "
                "пороге. Поэтому классы могут быть идеально упорядочены по score, "
                "но часть объектов может находиться по неправильную сторону "
                "фиксированного порога."
            )

    return "\n\n".join(paragraphs)


def build_conclusion(metrics):
    custom = metrics[metrics["model"] != REFERENCE_NAME].copy()
    reference = metrics[metrics["model"] == REFERENCE_NAME].copy()

    best_acc_value = custom["test_accuracy"].max()
    best_acc = custom[custom["test_accuracy"] == best_acc_value].copy()
    best_acc_names = ", ".join(
        f"`{name}`" for name in best_acc["model"].tolist()
    )

    best_auc_value = custom["roc_auc"].max()
    best_auc_names = ", ".join(
        f"`{name}`"
        for name in custom.loc[
            custom["roc_auc"] == best_auc_value, "model"
        ].tolist()
    )

    result = (
        "В лабораторной работе был реализован линейный бинарный классификатор "
        "с квадратичной функцией потерь. Реализованы вычисление и анализ "
        "отступов, градиент функции потерь, рекуррентная оценка функционала "
        "качества, SGD с momentum, L2-регуляризация, скорейший градиентный "
        "спуск, корреляционная и случайная инициализация весов, multistart, "
        "а также случайное предъявление объектов и предъявление по модулю "
        "отступа.\n\n"
        f"Максимальная test accuracy среди собственных реализаций равна "
        f"**{best_acc_value:.4f}** и получена для: {best_acc_names}. "
        f"Максимальный ROC AUC среди собственных реализаций равен "
        f"**{best_auc_value:.6f}** и получен для: {best_auc_names}."
    )

    random_row = custom[custom["model"] == "Correlation + random"]
    margin_row = custom[custom["model"] == "Correlation + |margin|"]
    if not random_row.empty and not margin_row.empty:
        rnd = random_row.iloc[0]
        mar = margin_row.iloc[0]
        result += (
            f"\n\nСлучайное предъявление и предъявление по модулю отступа "
            f"дали test accuracy **{rnd['test_accuracy']:.4f}** и "
            f"**{mar['test_accuracy']:.4f}**, F1 — "
            f"**{rnd['f1']:.4f}** и **{mar['f1']:.4f}**, "
            f"ROC AUC — **{rnd['roc_auc']:.6f}** и "
            f"**{mar['roc_auc']:.6f}** соответственно. "
            "Предъявление по модулю отступа немного улучшило accuracy и F1, "
            "при этом ROC AUC незначительно снизился."
        )

    if not reference.empty:
        ref = reference.iloc[0]
        best = best_acc.sort_values(
            ["roc_auc", "f1"], ascending=False
        ).iloc[0]
        result += (
            f"\n\nДля сравнения с эталоном выбрана `{best['model']}`. "
            f"Собственная реализация получила test accuracy "
            f"**{best['test_accuracy']:.4f}**, F1 = **{best['f1']:.4f}**, "
            f"ROC AUC = **{best['roc_auc']:.6f}**; эталонный "
            f"`SGDClassifier` — test accuracy **{ref['test_accuracy']:.4f}**, "
            f"F1 = **{ref['f1']:.4f}**, ROC AUC = **{ref['roc_auc']:.6f}**. "
            "Следовательно, собственная реализация SGD достигает качества, "
            "сопоставимого с библиотечным эталоном."
        )

    return result


def generate_report():
    check_inputs()
    metrics = pd.read_csv(METRICS_PATH)

    report = f"""# Лабораторная работа №1. Линейная классификация

## Цель работы

Реализовать линейный бинарный классификатор и основные элементы метода
стохастического градиента: вычисление отступа, квадратичную функцию потерь
и её градиент, рекуррентную оценку функционала качества, momentum,
L2-регуляризацию, скорейший градиентный спуск, различные способы
инициализации весов и предъявления объектов. Сравнить собственные реализации
с эталонным алгоритмом из `sklearn`.

## Датасет

Использован датасет **Banknote Authentication**. Он содержит 1372 объекта,
четыре вещественных признака и бинарную целевую переменную:

- `variance` — дисперсия;
- `skewness` — асимметрия;
- `curtosis` — эксцесс;
- `entropy` — энтропия.

Датасет не хранится в репозитории и загружается во время выполнения программы
из внешнего источника через `sklearn.datasets.fetch_openml`.

Два исходных класса преобразуются в $-1$ и $+1$, поскольку используемая
формула отступа предполагает

$$
y \\in \\{{-1,+1\\}}.
$$

В полном датасете 1372 объекта: 762 объекта одного класса и 610 объектов
другого класса, то есть примерно 55.5% и 44.5%. Таким образом, выраженного
дисбаланса классов нет.

Данные разделяются на train и test в отношении 80/20 со стратификацией,
поэтому соотношение классов сохраняется примерно одинаковым в обеих выборках.
Стандартизация выполняется без утечки данных: `StandardScaler` обучается
только на train-выборке, после чего это же преобразование применяется к test.

## Линейный классификатор и отступ

Для линейного классификатора используется дискриминантная функция

$$
g(x,w)=\\langle x,w\\rangle.
$$

Предсказание определяется знаком дискриминанта:

$$
a(x,w)=\\operatorname{{sign}}\\left(\\langle x,w\\rangle\\right).
$$

Отступ объекта определяется как

$$
M_i=y_i\\langle x_i,w\\rangle.
$$

Если $M_i<0$, объект классифицирован ошибочно. Если $M_i>0$, классификация
верна. Малое значение $|M_i|$ означает, что объект находится близко
к разделяющей границе.

Ниже показано распределение отступов **на обучающей выборке после обучения
модели `Correlation + |margin|`**. Объекты отсортированы по величине отступа;
пунктирная линия соответствует границе $M=0$.

![Распределение отступов на train для Correlation + |margin|](images/margins.png)

Большая часть объектов имеет положительный отступ, то есть классифицируется
верно. Небольшое число объектов имеет отрицательный отступ и классифицируется
ошибочно. Объекты с $|M_i|$, близким к нулю, находятся около разделяющей
границы и являются наиболее неуверенными для классификатора. Именно таким
объектам повышается вероятность предъявления при обучении по модулю отступа.

## Квадратичная функция потерь

Используется квадратичная функция потерь

$$
L(M)=(1-M)^2.
$$

Так как

$$
M=y\\langle x,w\\rangle,
$$

градиент функции потерь по весам равен

$$
\\nabla_w L
=
-2yx\\left(1-y\\langle x,w\\rangle\\right)
=
-2yx(1-M).
$$

## Рекуррентная оценка функционала качества

Начальное значение $Q$ рассчитывается как среднее значение функции потерь
на случайном подмножестве train-объектов. После обработки очередного объекта
используется рекуррентное обновление

$$
Q_t
=
\\lambda \\xi_t
+
(1-\\lambda)Q_{{t-1}},
$$

где $\\xi_t$ — значение функции потерь на объекте текущей итерации.

![Рекуррентная оценка функционала качества](images/q_history.png)

Поскольку разные способы предъявления объектов формируют разные
последовательности $\\xi_t$, дополнительно через одинаковые интервалы
вычисляется средний loss на всей train-выборке.

![Динамика среднего train loss](images/train_loss.png)

## Стохастический градиентный спуск с momentum

Для сглаживания стохастических градиентов используется momentum:

$$
v_t
=
\\gamma v_{{t-1}}
+
(1-\\gamma)\\nabla L_t,
$$

$$
w_t
=
w_{{t-1}}-h v_t.
$$

В эксперименте используются $\\gamma=0.9$ и $h=0.01$.

## L2-регуляризация

Регуляризованная функция потерь имеет вид

$$
\\widetilde{{L}}(w)
=
L(w)
+
\\frac{{\\tau}}{{2}}\\|w\\|^2.
$$

Её градиент:

$$
\\nabla \\widetilde{{L}}
=
\\nabla L
+
\\tau w.
$$

В эксперименте используется $\\tau=0.001$.

## Инициализация весов

### Корреляционная инициализация

Для каждого признака начальный вес вычисляется по формуле

$$
w_j
=
\\frac{{\\langle y,f_j\\rangle}}
{{\\langle f_j,f_j\\rangle}}.
$$

### Случайная инициализация и multistart

Начальные веса генерируются из распределения

$$
w_j
\\sim
U\\left(
-\\frac{{1}}{{2n}},
\\frac{{1}}{{2n}}
\\right).
$$

{build_multistart_text()}

![Сравнение запусков multistart](images/multistart.png)

## Предъявление объектов по модулю отступа

Помимо равновероятного случайного предъявления реализован вариант,
в котором объекты с малым абсолютным отступом предъявляются чаще:

$$
p_i
\\propto
\\frac{{1}}{{|M_i|+\\varepsilon}}.
$$

Чем меньше $|M_i|$, тем ближе объект к разделяющей границе и тем выше
вероятность его выбора.

## Скорейший градиентный спуск

Для одного объекта квадратичная функция потерь имеет вид

$$
L(w)
=
\\left(
1-y\\langle x,w\\rangle
\\right)^2.
$$

При используемом в программе градиенте

$$
\\nabla_w L
=
-2yx\\left(
1-y\\langle x,w\\rangle
\\right)
$$

используется шаг

$$
h^*
=
\\frac{{1}}{{2\\|x_i\\|^2}}.
$$

После этого веса обновляются по формуле

$$
w_{{t+1}}
=
w_t
-
h^*\\nabla_w L.
$$

Скорейший градиентный спуск исследуется отдельно от momentum и
L2-регуляризации, чтобы сравнить сам способ выбора шага.

## Эталонное решение

Для сравнения используется готовая реализация
`sklearn.linear_model.SGDClassifier`. Она применяется только как
эталонный алгоритм. Все элементы собственного линейного классификатора
реализованы отдельно с использованием `numpy`.

## Результаты

Итоговые значения метрик:

{build_results_table(metrics)}

![Сравнение качества моделей](images/metrics_comparison.png)

ROC-кривая строится по непрерывному значению дискриминанта

$$
s(x)=\\langle x,w\\rangle.
$$

Она показывает качество ранжирования объектов при изменении порога
классификации.

![ROC-кривые](images/roc_curves.png)

### Анализ результатов

{build_results_analysis(metrics)}

## Вывод

{build_conclusion(metrics)}
"""

    README_PATH.write_text(report, encoding="utf-8")
    print(f"Отчёт сохранён: {README_PATH}")


if __name__ == "__main__":
    generate_report()
