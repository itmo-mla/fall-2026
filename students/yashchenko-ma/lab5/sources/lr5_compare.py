"""Эквивалентность Ньютона-Рафсона, IRLS и эталона sklearn (без регуляризации).

Два набора данных: сбалансированная подвыборка 1:2 и полная выборка (3.4% отказов).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from common import setup_style, save_fig, C1, C2, C3, INK2
from logreg import LogisticRegressionIRLS, add_intercept, grad_hess, loss
from lr5_data import get_subsample_split, get_full_split, FEATURE_NAMES, save_csv

COEF_NAMES = ["intercept"] + FEATURE_NAMES


def sklearn_reference(X, y):
    """LogisticRegression без штрафа; точный solver newton-cg, жесткий tol."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in (dict(penalty=None), dict(C=np.inf)):
            try:
                return LogisticRegression(solver="newton-cg", tol=1e-12, max_iter=1000,
                                          **kw).fit(X, y)
            except Exception:
                continue
    raise RuntimeError("не удалось построить эталон sklearn")


def run(name, split):
    X_tr, X_te, y_tr, y_te = split
    print(f"\n{'=' * 70}\n{name}: train {X_tr.shape}, test {X_te.shape}, "
          f"доля +1 в train = {np.mean(y_tr == 1):.4f}\n{'=' * 70}")
    ms = {
        "newton": LogisticRegressionIRLS("newton").fit(X_tr, y_tr),
        "irls": LogisticRegressionIRLS("irls").fit(X_tr, y_tr),
        "irls_glm": LogisticRegressionIRLS("irls_glm").fit(X_tr, y_tr),
        "newton_zeros": LogisticRegressionIRLS("newton", init="zeros").fit(X_tr, y_tr),
    }
    sk = sklearn_reference(X_tr, y_tr)
    w_sk = np.concatenate([sk.intercept_, sk.coef_.ravel()])

    print("\nСходимость (tol = 1e-10 по max|dsigma|, init = МНК):")
    for k, m in ms.items():
        r = m.res_
        print(f"  {k:<13} итераций={r.n_iter:<3} сошёлся={r.converged}  Q={r.loss_hist[-1]:.6f}"
              f"  ||Q'||={r.grad_hist[-1]:.2e}")

    r = ms["newton"].res_
    print("\nТраектория Ньютона-Рафсона:")
    print(f"  {'t':>2} {'Q(w_t)':>14} {'||Q`(w_t)||':>14}")
    for t, (q, g) in enumerate(zip(r.loss_hist, r.grad_hist)):
        print(f"  {t:>2} {q:>14.6f} {g:>14.3e}")

    # --- Ньютон vs IRLS: итерация за итерацией ---
    def traj_diff(a, b):
        n = min(len(a.res_.w_hist), len(b.res_.w_hist))
        return np.array([np.abs(a.res_.w_hist[t] - b.res_.w_hist[t]).max() for t in range(n)])

    d_ni = traj_diff(ms["newton"], ms["irls"])
    d_ng = traj_diff(ms["newton"], ms["irls_glm"])
    print(f"\nmax|w_t(Newton) - w_t(IRLS)| по всем t:     {d_ni.max():.3e}")
    print(f"max|w_t(Newton) - w_t(IRLS-GLM)| по всем t: {d_ng.max():.3e}")
    print(f"max|w(Newton, init=МНК) - w(Newton, init=0)|: "
          f"{np.abs(ms['newton'].res_.w - ms['newton_zeros'].res_.w).max():.3e}"
          f"  (итераций из нуля: {ms['newton_zeros'].res_.n_iter})")

    # --- против sklearn ---
    w = ms["newton"].res_.w
    p_own, p_sk = ms["newton"].predict_proba(X_te), sk.predict_proba(X_te)[:, 1]
    y01_te = (y_te == 1).astype(int)
    print(f"\nпротив sklearn (newton-cg, penalty=None):")
    print(f"  max|w_own - w_sk|              = {np.abs(w - w_sk).max():.3e}")
    print(f"  max|w_irls - w_sk|             = {np.abs(ms['irls'].res_.w - w_sk).max():.3e}")
    print(f"  max|p_own - p_sk| (test)       = {np.abs(p_own - p_sk).max():.3e}")
    print(f"  |logloss_own - logloss_sk|     = "
          f"{abs(log_loss(y01_te, p_own) - log_loss(y01_te, p_sk)):.3e}")
    print(f"  совпадение меток (test)        = "
          f"{np.mean(ms['newton'].predict(X_te) == sk.predict(X_te)):.4f}")

    # --- выпуклость и условие оптимума ---
    F = add_intercept(X_tr)
    g, H = grad_hess(w, F, y_tr.astype(float))
    ev = np.linalg.eigvalsh(H)
    print(f"\nв оптимуме: ||Q'||={np.linalg.norm(g):.2e}, min eig Q''={ev.min():.3f} (>0: "
          f"строго выпукла), max eig={ev.max():.1f}, cond={ev.max() / ev.min():.1f}")

    # --- сл. 20: log-loss лекции == кросс-энтропия по вероятностям ---
    p_tr = expit(F @ w)
    y01 = (y_tr == 1).astype(float)
    ce = -np.sum(y01 * np.log(p_tr) + (1 - y01) * np.log(1 - p_tr))
    print(f"сл. 20: Q(w)={loss(w, F, y_tr.astype(float)):.9f}  vs  -ln L={ce:.9f}  "
          f"(|разн.|={abs(loss(w, F, y_tr.astype(float)) - ce):.2e})")

    print("\nВеса (Newton | sklearn | |разн.|):")
    for nm, a, b in zip(COEF_NAMES, w, w_sk):
        print(f"  {nm:<13}{a:>10.5f}{b:>10.5f}{abs(a - b):>11.2e}")

    # --- таблицы для CSV ---
    eq = {
        "n_iter_newton": ms["newton"].res_.n_iter,
        "n_iter_irls": ms["irls"].res_.n_iter,
        "n_iter_irls_glm": ms["irls_glm"].res_.n_iter,
        "n_iter_newton_zero_init": ms["newton_zeros"].res_.n_iter,
        "loss_Q_at_optimum": r.loss_hist[-1],
        "grad_norm_at_optimum": r.grad_hist[-1],
        "max_dw_newton_vs_irls_all_iterations": d_ni.max(),
        "max_dw_newton_vs_irls_glm_all_iterations": d_ng.max(),
        "max_dw_init_ols_vs_zeros": np.abs(w - ms["newton_zeros"].res_.w).max(),
        "max_dw_newton_vs_sklearn": np.abs(w - w_sk).max(),
        "max_dw_irls_vs_sklearn": np.abs(ms["irls"].res_.w - w_sk).max(),
        "max_dp_vs_sklearn_test": np.abs(p_own - p_sk).max(),
        "abs_dlogloss_vs_sklearn_test": abs(log_loss(y01_te, p_own) - log_loss(y01_te, p_sk)),
        "label_agreement_with_sklearn_test": float(np.mean(ms["newton"].predict(X_te) == sk.predict(X_te))),
        "abs_Q_minus_neg_loglik": abs(loss(w, F, y_tr.astype(float)) - ce),
        "hessian_min_eigenvalue": ev.min(),
        "hessian_max_eigenvalue": ev.max(),
    }
    traj = pd.DataFrame([
        dict(method=k, iteration=t, loss_Q=q, grad_norm=gn,
             **{f"w_{nm}": v for nm, v in zip(COEF_NAMES, wt)})
        for k, m in ms.items()
        for t, (q, gn, wt) in enumerate(zip(m.res_.loss_hist, m.res_.grad_hist, m.res_.w_hist))])
    weights = pd.DataFrame({
        "feature": COEF_NAMES, "newton": w, "irls": ms["irls"].res_.w,
        "irls_glm": ms["irls_glm"].res_.w, "sklearn": w_sk,
        "abs_diff_newton_sklearn": np.abs(w - w_sk)})

    return dict(ms=ms, w_sk=w_sk, p_own=p_own, p_sk=p_sk, wdiff=np.abs(w - w_sk),
                eq=eq, traj=traj, weights=weights)


def main():
    setup_style()
    res = {
        "Подвыборка 1:2": run("ПОДВЫБОРКА 1:2", get_subsample_split()),
        "Полная выборка": run("ПОЛНАЯ ВЫБОРКА", get_full_split()),
    }
    # --- CSV ---
    ids = {"Подвыборка 1:2": "subsample_1_2", "Полная выборка": "full"}

    def stack(key):
        df = pd.concat([r[key].assign(dataset=ids[n]) for n, r in res.items()], ignore_index=True)
        return df[["dataset"] + [c for c in df.columns if c != "dataset"]]

    save_csv(pd.DataFrame([dict(dataset=ids[n], quantity=k, value=v)
                           for n, r in res.items() for k, v in r["eq"].items()]),
             "lr5_equivalence.csv")
    save_csv(stack("traj"), "lr5_trajectory.csv")
    save_csv(stack("weights"), "lr5_weights.csv")

    cols = {"Подвыборка 1:2": C1, "Полная выборка": C2}

    # --- график 1: сходимость ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, r) in zip(axes, res.items()):
        for key, lab, col, mk, ms_, fill in (
                ("newton", "Ньютон-Рафсон", C1, "o", 7, C1),
                ("irls", "IRLS", C2, "s", 11, "none")):
            gh = r["ms"][key].res_.grad_hist
            ax.semilogy(range(len(gh)), np.maximum(gh, 1e-16), "-", color=col, marker=mk,
                        markersize=ms_, markerfacecolor=fill, label=lab)
        ax.set_title(name)
        ax.set_xlabel("итерация t")
        ax.legend()
    axes[0].set_ylabel("‖Q′(w_t)‖")
    fig.suptitle("Сходимость: траектории Ньютона-Рафсона и IRLS совпадают", y=1.02)
    save_fig(fig, "lr5_convergence.png")

    # --- график 2: эквивалентность с sklearn ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, r in res.items():
        a1.scatter(r["p_sk"], r["p_own"], s=10, alpha=0.5, color=cols[name], label=name)
    a1.plot([0, 1], [0, 1], "--", color=INK2, lw=1)
    a1.set_xlabel("sklearn: P(y=+1|x), test")
    a1.set_ylabel("Ньютон-Рафсон: P(y=+1|x), test")
    a1.set_title("Вероятности на тесте")
    a1.legend()
    x = np.arange(len(COEF_NAMES))
    for k, (name, r) in enumerate(res.items()):
        a2.bar(x + (k - 0.5) * 0.38, np.maximum(r["wdiff"], 1e-16), width=0.36,
               color=cols[name], label=name)
    a2.set_yscale("log")
    a2.set_xticks(x, COEF_NAMES, rotation=35, ha="right")
    a2.set_ylabel("|w_own − w_sklearn|")
    a2.set_title("Расхождение весов с sklearn")
    a2.set_ylim(8e-17, 6e-14)
    a2.legend(ncol=2, loc="upper right")
    save_fig(fig, "lr5_vs_sklearn.png")


if __name__ == "__main__":
    main()
