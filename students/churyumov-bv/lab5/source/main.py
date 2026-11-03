# 1. выбрать датасет для логистической регрессии;
# 2. реализовать логистическую регрессию с помощью метода Ньютона-Рафсона и IRLS;
# 3. показать эквивалентность с эталонной реализацией;
# 4. подготовить отчет.

import os
import warnings
import numpy as np
import logreg
import plots
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
GD_ITERATIONS = 1000

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def path(filename):
    return os.path.join(OUTPUT_DIR, filename)


# 1. выбрать датасет для логистической регрессии;
def load_data():
    data = fetch_openml("MagicTelescope", version=1, as_frame=True, parser="auto")  # MAGIC Gamma Telescope
    X = data.data
    names = [c.rstrip(":") for c in X.columns]                     # в OpenML имена признаков заканчиваются двоеточием
    y01 = (data.target.astype(str) == "g").values.astype(int)      # 1 - гамма-квант, 0 - адрон (фон)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(float), y01, test_size=0.2, random_state=RANDOM_STATE, stratify=y01)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, names


def metrics(y01, p):
    return accuracy_score(y01, p > 0.5), log_loss(y01, p), roc_auc_score(y01, p)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
   
    X_train, X_test, y01_train, y01_test, names = load_data()
    F, F_test = logreg.add_bias(X_train), logreg.add_bias(X_test)
    y, y_test = 2 * y01_train - 1, 2 * y01_test - 1  # метки в {-1, +1}, как на слайдах 6-9
    p = F.shape[1]
    
    print(f"train: {X_train.shape}, test: {X_test.shape}; доля класса 1: train {y01_train.mean():.3f}, test {y01_test.mean():.3f}")

    # 2. реализовать логистическую регрессию с помощью метода Ньютона-Рафсона и IRLS;
    w_ols = logreg.least_squares(F, y)  # нулевое приближение IRLS
    w_zero = np.zeros(p)
    print("\nМетоды второго порядка (w⁰ = МНК): итераций, Q(w*), max|∂Q/∂w|")
    hists = {
        "Ньютон–Рафсон": logreg.newton_raphson(F, y, w_ols),
        "IRLS (слайд 9)": logreg.irls(F, y, w_ols),
        "IRLS для GLM (слайд 19)": logreg.irls_glm(F, y01_train, w_ols),
    }
    for name, h in hists.items():
        print(f"{name:24s} | {h.n_iter:2d} | {h.loss[-1]:.10f} | {h.grad_norm[-1]:.2e}")
    newton, irls, glm = hists.values()

    print("\nЗависимость от нулевого приближения (итераций до остановки):")
    for label, w0 in (("w⁰ = МНК", w_ols), ("w⁰ = 0", w_zero)):
        print(f"{label:9s} | Ньютон {logreg.newton_raphson(F, y, w0).n_iter}, IRLS {logreg.irls(F, y, w0).n_iter}, "
              f"IRLS-GLM {logreg.irls_glm(F, y01_train, w0).n_iter}")

    # эквивалентность методов между собой: поитерационно
    print("\nСовпадение траекторий (w⁰ = МНК): max_t max_j |wⱼᵗ_Ньютон − wⱼᵗ_метод|")
    print(f"Ньютон vs IRLS       = {max(np.abs(a - b).max() for a, b in zip(newton.w, irls.w)):.2e}")
    print(f"Ньютон vs IRLS-GLM   = {max(np.abs(a - b).max() for a, b in zip(newton.w, glm.w)):.2e}")
    print("Норма градиента по итерациям (Ньютон):", " ".join(f"{g:.1e}" for g in newton.grad_norm))
    print(f"Q(w^t) - Q(w*) по итерациям (Ньютон):  ", " ".join(f"{v - newton.loss[-1]:.1e}" for v in newton.loss))

    w_star = newton.w_star
    q_star = newton.loss[-1]
    w_gd, gd_loss = logreg.gradient_descent(F, y, w_ols, GD_ITERATIONS)
    gd_gap = gd_loss - q_star
    n_gd = f"за {int(np.argmax(gd_gap < 1e-10))} итераций" if (gd_gap < 1e-10).any() else f"не достигнуто за {GD_ITERATIONS} итераций"
    print(f"\nГрадиентный спуск (шаг 1/L): Q − Q* = {gd_gap[10]:.1e} после 10, {gd_gap[50]:.1e} после 50, "
          f"{gd_gap[200]:.1e} после 200, {gd_gap[-1]:.1e} после {GD_ITERATIONS} итераций; Q − Q* < 1e-10 {n_gd} "
          f"(Ньютон: за {int(np.argmax(np.array(newton.loss) - q_star < 1e-10))})")
    plots.plot_convergence(hists, gd_loss, q_star, path("convergence.png"))

    # 3. показать эквивалентность с эталонной реализацией;
    p_own = logreg.sigmoid(F_test @ w_star)
    
    print("\nСравнение со sklearn.linear_model.LogisticRegression(penalty=None):")
    print("solver     | max|w_own − w_sk| | max|p_own − p_sk| (test) | accuracy | log-loss | AUC")
   
    ref = {}
    for solver in ("newton-cg", "lbfgs"):
        sk = LogisticRegression(penalty=None, solver=solver, tol=1e-12, max_iter=10000).fit(X_train, y01_train)
        w_sk = np.concatenate([sk.intercept_, sk.coef_[0]])
        p_sk = sk.predict_proba(X_test)[:, 1]
        ref[solver] = (w_sk, p_sk)
        print(f"{solver:10s} | {np.abs(w_star - w_sk).max():17.2e} | {np.abs(p_own - p_sk).max():24.2e} | "
              f"{'%.4f / %.4f' % (accuracy_score(y01_test, p_sk > 0.5), accuracy_score(y01_test, p_own > 0.5))} | "
              f"{'%.4f / %.4f' % (log_loss(y01_test, p_sk), log_loss(y01_test, p_own))} | "
              f"{'%.4f / %.4f' % (roc_auc_score(y01_test, p_sk), roc_auc_score(y01_test, p_own))}   (sklearn / своя)")
    sk_l2 = LogisticRegression(solver="newton-cg", tol=1e-12).fit(X_train, y01_train)  # настройки по умолчанию: L2, C = 1
    
    w_l2 = np.concatenate([sk_l2.intercept_, sk_l2.coef_[0]])
    
    print(f"sklearn по умолчанию (L2, C=1): max|w_own − w_sk| = {np.abs(w_star - w_l2).max():.2e}, "
          f"max|p_own − p_sk| = {np.abs(p_own - sk_l2.predict_proba(X_test)[:, 1]).max():.2e}")
    plots.plot_equivalence(["свободный"] + names, w_star, ref["newton-cg"][0], p_own, ref["newton-cg"][1], path("equivalence.png"))

    # интерпретация коэффициентов: P/(1-P) умножается на e^{w_j} при росте признака на одно СКО (слайд 21)
    se = np.sqrt(np.diag(np.linalg.inv(logreg.hessian(F, y, w_star))))  # гессиан Q в оптимуме - информационная матрица Фишера
    
    print("\nКоэффициенты (признаки стандартизованы):")
   
    print("признак  |      wⱼ |  SE(wⱼ) |      z | отношение шансов e^wⱼ")
    
    for j, name in enumerate(["свободный"] + names):
        print(f"{name:8s} | {w_star[j]:+7.3f} | {se[j]:7.3f} | {w_star[j] / se[j]:+6.2f} | {np.exp(w_star[j]):.3f}")

    # метрики качества
    print("\nКачество на test: accuracy | log-loss | AUC")
    
    base = max(y01_test.mean(), 1 - y01_test.mean())
    
    print(f"логистическая регрессия: {metrics(y01_test, p_own)[0]:.4f} | {metrics(y01_test, p_own)[1]:.4f} | {metrics(y01_test, p_own)[2]:.4f}")
    print(f"константа (мажоритарный класс): accuracy {base:.4f}")
    
    p_train = logreg.sigmoid(F @ w_star)
    
    print(f"train: accuracy {accuracy_score(y01_train, p_train > 0.5):.4f}, log-loss {log_loss(y01_train, p_train):.4f}, AUC {roc_auc_score(y01_train, p_train):.4f}")

    # вопрос 4: МНК как классификатор на ответах {0, 1}
    w_lin = logreg.least_squares(F, y01_train.astype(float))
    z_lin = F_test @ w_lin
    outside = np.mean((z_lin < 0) | (z_lin > 1))
    
    print(f"\nМНК на ответах {{0, 1}}: accuracy test {accuracy_score(y01_test, z_lin > 0.5):.4f}, AUC {roc_auc_score(y01_test, z_lin):.4f}, "
          f"доля «вероятностей» вне [0, 1]: {outside:.3f} (диапазон {z_lin.min():.2f} … {z_lin.max():.2f})")
    plots.plot_ols(p_own, z_lin, y01_test, path("ols.png"))

    M0, M = logreg.margins(F, y, w_ols), logreg.margins(F, y, w_star)
    weight = lambda M: logreg.sigmoid(M) * (1 - logreg.sigmoid(M))
    
    print(f"\nВеса (1−σ)σ при w*: max {weight(M).max():.4f} (при M = 0), min {weight(M).min():.2e}; "
          f"средний вес правильно/неправильно классифицированных: {weight(M)[M > 0].mean():.4f} / {weight(M)[M < 0].mean():.4f}")
    
    print(f"Доля объектов с отступом M<0: w⁰ (МНК) {np.mean(M0 < 0):.3f}, w* {np.mean(M < 0):.3f}; "
          f"максимум |ỹ| = e^(−M/2): w⁰ {np.exp(-M0 / 2).max():.2f}, w* {np.exp(-M / 2).max():.2f}")
    plots.plot_weights(F, y, w_ols, w_star, path("weights.png"))

    Xb, yb = load_breast_cancer(return_X_y=True)
    Xb_train, _, yb_train, _ = train_test_split(Xb, yb, test_size=0.2, random_state=RANDOM_STATE, stratify=yb)
    Fb = logreg.add_bias(StandardScaler().fit_transform(Xb_train))
    h_sep = logreg.newton_raphson(Fb, 2 * yb_train - 1, np.zeros(Fb.shape[1]), tol=0, max_iter=30)
    acc_sep = accuracy_score(yb_train, Fb @ h_sep.w_star > 0)
    
    print(f"\nBreast Cancer (разделимая train): Q(w^t) = {h_sep.loss[1]:.3f} (t=1), {h_sep.loss[5]:.2e} (t=5), {h_sep.loss[-1]:.2e} (t=30); "
          f"‖w^t‖ = {np.linalg.norm(h_sep.w[1]):.1f}, {np.linalg.norm(h_sep.w[5]):.1f}, {np.linalg.norm(h_sep.w[10]):.1f}, "
          f"{np.linalg.norm(h_sep.w[20]):.1f}, {np.linalg.norm(h_sep.w[30]):.1f}; accuracy train {acc_sep:.4f}")
    
    print(f"MAGIC для сравнения (w⁰ = 0): ‖w*‖ = {np.linalg.norm(w_star):.3f}")
    h_magic = logreg.newton_raphson(F, y, w_zero, tol=0, max_iter=30)
    
    print(f"MAGIC: ‖w^t‖ при t = 10, 30: {np.linalg.norm(h_magic.w[10]):.4f}, {np.linalg.norm(h_magic.w[30]):.4f}")
    plots.plot_separable(h_magic, h_sep, path("separable.png"))
