import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score

import data
import logistic as lg
import plots


def banner(t):
    print(f"\n=== {t} ===")


def newton_table(h):
    print(f"{'t':>3} {'Q(w)':>14} {'||grad||':>11} {'||шаг||':>11} {'шаг_t/шаг_(t-1)^2':>18}")
    for t in range(len(h.w)):
        s = h.step_norm[t] if t < len(h.step_norm) else float("nan")
        ok = 0 < t < len(h.step_norm) and h.step_norm[t - 1] > 1e-5
        ratio = h.step_norm[t] / h.step_norm[t - 1] ** 2 if ok else float("nan")
        print(f"{t:>3} {h.loss[t]:>14.6f} {h.grad_norm[t]:>11.2e} {s:>11.2e} {ratio:>18.3f}")


def sk_weights(m):
    return np.concatenate([m.intercept_, m.coef_.ravel()])


def calibration_bins(p, y, n=6):
    parts = np.array_split(np.argsort(p), n)
    return [p[i].mean() for i in parts], [np.mean(y[i] == 1) for i in parts], [len(i) for i in parts]


def part_pima():
    X, y, names = data.load_pima()
    Xtr, Xte, ytr, yte = data.split(X, y)
    F, Fte = data.preprocess(Xtr, Xte, names)
    yb = (ytr + 1) // 2
    yb_te = (yte + 1) // 2
    print(f"Pima: {X.shape[0]} объектов, {X.shape[1]} признаков; train {len(ytr)} (доля +1: {np.mean(ytr == 1):.3f}), "
          f"test {len(yte)} (доля +1: {np.mean(yte == 1):.3f})")
    print(f"число обусловленности F: {np.linalg.cond(F):.2f}")
    print("нули-пропуски в train:", {n: int((Xtr[:, names.index(n)] == 0).sum()) for n in data.MISSING_AS_ZERO})

    banner("Ньютон-Рафсон, tau = 0, w0 = 0")
    hn = lg.newton_raphson(F, ytr)
    newton_table(hn)
    w_star = hn.w_star

    banner("IRLS (маржинальная запись) и IRLS в записи GLM")
    hi, hg = lg.irls(F, ytr), lg.irls_glm(F, ytr)
    print(f"итераций: Ньютон {hn.n_iter}, IRLS {hi.n_iter}, GLM {hg.n_iter}")
    print(f"{'t':>3} {'max|Ньютон-IRLS|':>18} {'max|Ньютон-GLM|':>17}")
    for t in range(min(len(hn.w), len(hi.w), len(hg.w))):
        print(f"{t:>3} {np.abs(hn.w[t] - hi.w[t]).max():>18.2e} {np.abs(hn.w[t] - hg.w[t]).max():>17.2e}")
    M = lg.margins(F, ytr, w_star)
    g2 = expit(M) * (1 - expit(M))
    print(f"веса gamma_i^2 в решении: max {g2.max():.3f}, min {g2.min():.2e}, сумма {g2.sum():.1f} из {len(ytr)}")
    print(f"отступы трёх самых весомых объектов: {np.round(M[np.argsort(-g2)[:3]], 3)}; "
          f"трёх самых лёгких: {np.round(M[np.argsort(g2)[:3]], 2)}")
    w_ls = lg.least_squares(F, ytr)
    h_ls = lg.irls(F, ytr, w0=w_ls)
    print(f"IRLS из МНК-приближения: {h_ls.n_iter} итераций, max|w - w*| = {np.abs(h_ls.w_star - w_star).max():.2e}")

    banner("Сравнение с sklearn (penalty=None)")
    Xs, Xs_te = F[:, 1:], Fte[:, 1:]
    refs = {
        "newton-cholesky": LogisticRegression(penalty=None, solver="newton-cholesky", tol=1e-12, max_iter=200),
        "lbfgs": LogisticRegression(penalty=None, solver="lbfgs", tol=1e-12, max_iter=10000),
    }
    p_own = lg.probability(Fte, w_star)
    for nm, m in refs.items():
        m.fit(Xs, yb)
        pr = m.predict_proba(Xs_te)[:, 1]
        print(f"{nm:>16}: max|dw| = {np.abs(sk_weights(m) - w_star).max():.2e}, max|dP| = {np.abs(pr - p_own).max():.2e}, "
              f"совпадение предсказаний {np.mean((pr >= .5) == (p_own >= .5)) * 100:.1f}%, итераций sklearn {int(np.max(m.n_iter_))}")
    w_ref = sk_weights(refs["newton-cholesky"])
    print(f"Q(w_own) = {lg.log_loss(F, ytr, w_star):.8f}, Q(w_sklearn) = {lg.log_loss(F, ytr, w_ref):.8f}")

    banner("Качество на тесте")
    rows = {"Ньютон-Рафсон": w_star, "IRLS": hi.w_star, "IRLS (GLM)": hg.w_star, "sklearn newton-cholesky": w_ref}
    print(f"{'модель':>24} {'accuracy':>9} {'log-loss':>9} {'AUC':>7}")
    for nm, w in rows.items():
        p = lg.probability(Fte, w)
        print(f"{nm:>24} {lg.accuracy(yte, lg.predict(Fte, w)):>9.4f} {sk_log_loss(yb_te, p):>9.4f} {lg.auc(yte, p):>7.4f}")
    s_ls = Fte @ w_ls
    print(f"{'МНК на метках':>24} {lg.accuracy(yte, lg.predict(Fte, w_ls)):>9.4f} {'-':>9} {lg.auc(yte, s_ls):>7.4f}")
    print(f"AUC собственный {lg.auc(yte, p_own):.6f} против sklearn.roc_auc_score {roc_auc_score(yte, p_own):.6f}")
    print(f"оценки МНК на тесте лежат в [{s_ls.min():.2f}, {s_ls.max():.2f}] - вероятностью не являются")

    banner("L2-регуляризация: tau = 1 <-> C = 1 в sklearn")
    hr, hri = lg.newton_raphson(F, ytr, tau=1.0), lg.irls(F, ytr, tau=1.0)
    w_r = sk_weights(LogisticRegression(C=1.0, solver="newton-cholesky", tol=1e-12).fit(Xs, yb))
    print(f"Ньютон {hr.n_iter} ит., IRLS {hri.n_iter} ит.; max|dw| Ньютон-sklearn {np.abs(hr.w_star - w_r).max():.2e}, "
          f"IRLS-sklearn {np.abs(hri.w_star - w_r).max():.2e}")

    banner("Скорость сходимости")
    _, loss_gd = lg.gradient_descent(F, ytr, n_iter=300)
    q_star = hn.loss[-1]
    gap = loss_gd - q_star
    first = lambda tol: int(np.argmax(gap < tol)) if (gap < tol).any() else None
    print(f"итераций до Q - Q* < 1e-6: Ньютон {int(np.argmax(np.array(hn.loss) - q_star < 1e-6))}, "
          f"IRLS {int(np.argmax(np.array(hi.loss) - q_star < 1e-6))}, градиентный спуск {first(1e-6)}")
    print(f"градиентный спуск, 300 итераций: Q - Q* = {gap[-1]:.2e}; Ньютон сошёлся за {hn.n_iter}")
    plots.convergence({"Ньютон-Рафсон": hn, "IRLS": hi}, loss_gd, q_star)

    banner("Веса и интерпретация")
    print(f"{'признак':>8} {'w':>8} {'sklearn':>9} {'exp(w)':>8}")
    for j, nm in enumerate(["bias"] + names):
        print(f"{nm:>8} {w_star[j]:>8.4f} {w_ref[j]:>9.4f} {np.exp(w_star[j]):>8.3f}")
    plots.odds(names, w_star[1:], w_ref[1:])
    plots.calibration(*calibration_bins(p_own, yte))

    banner("Двумерная модель (plas, mass)")
    cols = [0] + [names.index("plas") + 1, names.index("mass") + 1]
    h2 = lg.newton_raphson(F[:, cols], ytr)
    print(f"w = {np.round(h2.w_star, 4)}, accuracy на тесте {lg.accuracy(yte, lg.predict(Fte[:, cols], h2.w_star)):.4f}")
    plots.boundary(F[:, cols], ytr, h2.w_star, ["plas", "mass"])


def part_separable():
    banner("Линейно разделимая выборка (Breast Cancer): tau = 0 и tau > 0")
    X, y, _ = data.load_breast_cancer_signed()
    Xtr, Xte, ytr, yte = data.split(X, y)
    F, Fte = data.preprocess(Xtr, Xte)
    hb = lg.newton_raphson(F, ytr, tol=0.0, max_iter=25)
    print(f"accuracy на train при tau = 0: {lg.accuracy(ytr, lg.predict(F, hb.w_star)):.4f}")
    norms = [np.linalg.norm(w) for w in hb.w]
    print("||w|| на итерациях 0,4,8,...:", [round(float(n), 1) for n in norms[::4]], f"; Q на конце {hb.loss[-1]:.2e}")
    plots.separable(norms, hb.loss)
    print(f"{'tau':>6} {'ит.':>4} {'||w||':>9} {'Q без штрафа':>13} {'acc test':>9}")
    for tau in [0.01, 0.1, 1, 10, 100]:
        h = lg.newton_raphson(F, ytr, tau=tau)
        w = h.w_star
        print(f"{tau:>6} {h.n_iter:>4} {np.linalg.norm(w):>9.2f} {lg.log_loss(F, ytr, w):>13.3f} {lg.accuracy(yte, lg.predict(Fte, w)):>9.4f}")


def main():
    part_pima()
    part_separable()
    print("\nГрафики сохранены в директорию lab5.")


if __name__ == "__main__":
    main()
