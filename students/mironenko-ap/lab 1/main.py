"""Лабораторная работа: линейный классификатор, квадратичная потеря + L2.

Матрица конфигураций:
    optimizer ∈ {momentum, steepest}
    init      ∈ {random, correlation, multistart}
    ordering  ∈ {random, margin}  — ТОЛЬКО для momentum;
               для steepest ordering не влияет (полнобатчевый),
               поэтому запускается один раз с фиксированным "random".
    L2        ∈ {on (lam=1e-3), off (lam=0)}

Плюс эталон sklearn: SGDClassifier.
"""
import numpy as np
from pathlib import Path

from src.data import load_data
from src.core import (
    loss_only, margin, check_gradient, EMA, log, clear_log, dump_log,
)
from src.init import init_random, init_correlation, multistart
from src.metrics import report, print_report
from src.optim import sgd_momentum, steepest_descent
from src.plots import (
    plot_margin_hist, plot_loss_curves, plot_ema_curves,
    plot_confusion, plot_decision_boundary, plot_bar_by_axis,
)
from src.reference import train_logreg, sgd_classifier_train

#  Гиперпараметры
LAM = 1e-3
LR = 5e-3
LR_DECAY = 0.05
MU = 0.8
EPOCHS = 30
#EARLY_STOP_DELTA = 1e-3
EARLY_STOP_PATIENCE = 3
SEED = 42
MULTISTART_N = 5
PRINT_EVERY = 5
EMA_ALPHAS = (0.9, 0.99, 0.999)

LOG_DIR = Path(__file__).resolve().parent / "results" / "logs" #директория для логов
LOG_DIR.mkdir(parents=True, exist_ok=True) #создать если отсутствует
REPORT_PATH = LOG_DIR / "report.txt"

OPTIMIZERS = ("momentum", "steepest")   #коллекция оптимизаторов
INITS = ("correlation", "multistart")   #коллекция инициализаций весов
ORDERINGS = ("random", "margin")
L2_VARIANTS = (("on", LAM, True), ("off", LAM, False)) # переключатель L2 со значением

#Для steepest ordering — no-op, фиксируем значение-заглушку.
ORDERING_NOOP = "—"             # отображается в таблице и на графике
ORDERING_STEEPEST = "random"    # что реально передаётся в оптимизатор

""" Вспомогательные функции"""
def step(n, title):
    log(f"ШАГ {n}: {title}")


def banner(text):
    log(text)


# 0. Данные
clear_log() #для начала очистим лог
log("Загрузка и подготовка данных...")
data = load_data()
X_tr, y_tr = data["X_train"], data["y_train"] #тренировочная выборка
X_val, y_val = data["X_val"], data["y_val"] #валидационная выборка
X_te, y_te = data["X_test"], data["y_test"] #тестовая выборка
d = X_tr.shape[1]
log(f"[info] d={d}, N_train={len(y_tr)}, N_val={len(y_val)}, N_test={len(y_te)}")


def _pred(w, b, X):
    return np.where(X @ w + b >= 0, 1, -1)


def _fit_once(optimizer, w0, b0, X, y, X_val, y_val,
              ordering, use_l2, verbose, seed):
    if optimizer == "momentum":
        return sgd_momentum(
            X, y, w0, b0,
            lr=LR, mu=MU, use_l2=use_l2,
            lr_decay=LR_DECAY,
            epochs=EPOCHS, ordering=ordering,
            rng=np.random.default_rng(seed),
            X_val=X_val, y_val=y_val,
            print_every=PRINT_EVERY, verbose=verbose,
            ema_alphas=EMA_ALPHAS,
            early_stop_delta=5e-2,
            early_stop_patience=EARLY_STOP_PATIENCE,
        )
    return steepest_descent(
        X, y, w0, b0,
        use_l2=use_l2,
        epochs=EPOCHS,
        X_val=X_val, y_val=y_val,
        print_every=PRINT_EVERY, verbose=verbose,
        ema_alphas=EMA_ALPHAS,
        early_stop_delta=1e-3,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )


def run_experiment(optimizer, init_kind, ordering, use_l2, seed=SEED):
    if init_kind == "random":
        w0, b0 = init_random(d, np.random.default_rng(seed))
        return _fit_once(optimizer, w0, b0, X_tr, y_tr, X_val, y_val,
                         ordering, lam, use_l2, True, seed)
    if init_kind == "correlation":
        w0, b0 = init_correlation(X_tr, y_tr)
        return _fit_once(optimizer, w0, b0, X_tr, y_tr, X_val, y_val,
                         ordering, use_l2, True, seed)
    if init_kind == "multistart":
        def fit_for_ms(w0, b0, X, y, X_val=None, y_val=None, **kw):
            return _fit_once(optimizer, w0, b0, X, y, X_val, y_val,
                             ordering, use_l2, False, seed)

        return multistart(
            fit_for_ms, X_tr, y_tr, X_val, y_val,
            n_starts=MULTISTART_N, seed=seed, lam=lam, use_l2=use_l2,
        )
    raise ValueError(f"init_kind={init_kind}")



# 1. Отступ
step(1, "Отступ объекта (случайная инициализация)")
w0, b0 = init_random(d, np.random.default_rng(SEED)) #рандом для визуализации отступов
m = margin(X_tr, y_tr, w0, b0)
log(f"  средний отступ : {m.mean():.4f}")
log(f"  std            : {m.std():.4f}")
log(f"  min / max      : {m.min():.4f} / {m.max():.4f}")
log(f"  доля m < 0     : {np.mean(m < 0):.4f}")
log(f"  доля m < 1     : {np.mean(m < 1):.4f}")
plot_margin_hist(X_tr, y_tr, w0, b0, name="step1_margin_before.png")


# 2. Проверка градиента
step(2, "Проверка градиента (конечные разности)")
check_gradient(X_tr, y_tr, w0, b0, LAM)


# 3. EMA — демонстрация
step(3, "Рекуррентная оценка (EMA): демонстрация")
for a in EMA_ALPHAS:
    ema = EMA(alpha=a)
    seq = [1.0, 0.8, 0.6, 0.5, 0.4]
    vals = [ema.update(x) for x in seq]
    log(f"  α={a}: {['%.4f' % v for v in vals]}")


# 4. Матрица конфигураций
step(4, "Матрица конфигураций: optimizer × init × ordering × L2")
log("  Примечание: для optimizer=steepest ordering не влияет "
    "(полнобатчевый метод) — запускается один раз.")

results = {}
histories = {}

for opt in OPTIMIZERS:
    #Для steepest перебираем только одно значение ordering.
    ordering_choices = ORDERINGS if opt == "momentum" else (ORDERING_STEEPEST,)

    for init_kind in INITS:
        for ordering in ordering_choices:
            for l2_tag, lam, use_l2 in L2_VARIANTS:
                banner(f"opt={opt} | init={init_kind} "
                       f"| ordering={ordering} | L2={l2_tag}")
                w, b, hist = run_experiment(
                    opt, init_kind, ordering, use_l2, seed=SEED
                )

                # Валидационные потери: полная цель и «чистая» data-потеря.
                val_loss = loss_only(X_val, y_val, w, b,
                                     lam if use_l2 else 0.0)
                val_data_loss = loss_only(X_val, y_val, w, b, 0.0)

                pred_te = _pred(w, b, X_te)
                scores_te = X_te @ w + b
                rep_te = report(y_te, pred_te, scores_te)

                # Для steepest в ключе — "-", чтобы не путать
                # с реальным "random" у momentum.
                ordering_key = ordering if opt == "momentum" else ORDERING_NOOP
                key = (opt, init_kind, ordering_key, l2_tag)
                results[key] = dict(
                    w=w, b=b, lam=lam, use_l2=use_l2,
                    val_loss=val_loss,
                    val_data_loss=val_data_loss,
                    test_rep=rep_te,
                )
                histories[key] = hist
                log(f"  val_loss = {val_loss:.4f} "
                    f"| val_data_loss = {val_data_loss:.4f} "
                    f"| test_acc = {rep_te['accuracy']:.4f} "
                    f"| ROC-AUC = {rep_te['roc_auc']:.4f}")

log(f"\n  Всего конфигураций: {len(results)}")


# 5. Эталон sklearn
step(5, "Эталон sklearn")
w0_ref, b0_ref = init_random(d)
w_ref_sq, b_ref_sq, hist_ref_sq = sgd_classifier_train(
    X_tr, y_tr, w0_ref, b0_ref,
    loss="squared_error", lam=LAM, use_l2=True,
    lr=LR, mu=MU, epochs=EPOCHS,
    ordering="random",
    X_val=X_val, y_val=y_val,
    print_every=PRINT_EVERY, verbose=True,
    ema_alphas=EMA_ALPHAS,
)
rep_ref_sq = report(
    y_te,
    _pred(w_ref_sq, b_ref_sq, X_te),
    X_te @ w_ref_sq + b_ref_sq,
)
val_ref_sq = loss_only(X_val, y_val, w_ref_sq, b_ref_sq, 0.0)
print_report("SGD-sklearn (squared_error)", rep_ref_sq)

#Отступ после обучения
m_b = margin(X_te, y_te, w_ref_sq, b_ref_sq)
log(f"  Эталон. средний отступ : {m_b.mean():.4f}")
log(f"  Эталон. std            : {m_b.std():.4f}")
log(f"  Эталон. min / max      : {m_b.min():.4f} / {m_b.max():.4f}")
log(f"  Эталон. доля m < 0     : {np.mean(m_b < 0):.4f}")
log(f"  Эталон. доля m < 1     : {np.mean(m_b < 1):.4f}")
plot_margin_hist(X_te, y_te, w_ref_sq, b_ref_sq, name="margin_after_etalon.png")


# 6. Сводная таблица
step(6, f"Сводная таблица ({len(results)} своих + 2 эталона)")

header = (f"\n{'optimizer':10s} {'init':12s} {'ordering':9s} {'L2':4s} "
          f"{'val_loss':>10s} {'test_acc':>10s} {'roc_auc':>10s} "
          f"{'f1':>8s} {'||w||':>7s}")
log(header)
log("")

for (o, i, r, l2), res in results.items():
    rep = res["test_rep"]
    wnorm = float(np.linalg.norm(res["w"]))
    log(f"{o:10s} {i:12s} {r:9s} {l2:4s} "
        f"{res['val_loss']:10.4f} {rep['accuracy']:10.4f} "
        f"{rep['roc_auc']:10.4f} {rep['f1']:8.4f} {wnorm:7.4f}")

log("")
log(f"{'SGD-sq':10s} {'zeros':12s} {'-':9s} {'on':4s} "
    f"{val_ref_sq:10.4f} {rep_ref_sq['accuracy']:10.4f} "
    f"{rep_ref_sq['roc_auc']:10.4f} {rep_ref_sq['f1']:8.4f} "
    f"{np.linalg.norm(w_ref_sq):7.4f}")

#best_key = min(results, key=lambda k: results[k]["val_data_loss"])
best_key = min(results, key=lambda k: results[k]["val_data_loss"])
best = results[best_key]
log(f"\nЛучшая конфигурация по val_data_loss: {best_key}")
log(f"  val_loss      = {best['val_loss']:.4f}")
log(f"  val_data_loss = {best['val_data_loss']:.4f}")
log(f"  test_acc      = {best['test_rep']['accuracy']:.4f}")
log(f"  ROC-AUC       = {best['test_rep']['roc_auc']:.4f}")
log(f"  F1            = {best['test_rep']['f1']:.4f}")

#Отступ после обучения
w_b, b_b = best["w"], best["b"]
m_b = margin(X_te, y_te, w_b, b_b)
log(f"  Best. средний отступ : {m_b.mean():.4f}")
log(f"  Best. std            : {m_b.std():.4f}")
log(f"  Best. min / max      : {m_b.min():.4f} / {m_b.max():.4f}")
log(f"  Best. доля m < 0     : {np.mean(m_b < 0):.4f}")
log(f"  Best. доля m < 1     : {np.mean(m_b < 1):.4f}")
plot_margin_hist(X_te, y_te, w_b, b_b, name="margin_after_fit.png")


# 7. Графики
step(7, "Графики")
plot_ema_curves(histories[best_key], name="step3_ema_vs_loss.png", alphas=EMA_ALPHAS)
cmp_keys = [
    ("momentum", "correlation", "random", "on"),
    ("momentum", "multistart",  "random", "on"),
    ("momentum", "correlation", "margin", "on"),
    ("momentum", "multistart",  "margin", "on"),
    ("steepest", "correlation", "—",      "on"),
    ("steepest", "multistart",  "—",      "on"),
]
plot_loss_curves(
    {f"{o}, {i}, {r}, L2={l}": histories[(o, i, r, l)]
     for (o, i, r, l) in cmp_keys if (o, i, r, l) in histories},
    name="step6_loss_curves.png",
)

plot_confusion(y_te, _pred(best["w"], best["b"], X_te), name="confusion.png")
plot_decision_boundary(X_te, y_te, best["w"], best["b"], name="boundary_best.png")

plot_bar_by_axis(results, axis_idx=0, axis_label="optimizer", name="axis_optimizer.png")
plot_bar_by_axis(results, axis_idx=1, axis_label="init", name="axis_init.png")
plot_bar_by_axis(results, axis_idx=2, axis_label="ordering (у steepest — no-op)",
                 name="axis_ordering.png")
plot_bar_by_axis(results, axis_idx=3, axis_label="L2", name="axis_L2.png")


# 8. Сохранение отчёта
step(8, "Сохранение отчёта")
dump_log(REPORT_PATH)
