"""Линейная регрессия через SVD: МНК (слайд 8), ридж (слайды 11-14), PCR (слайды 24-25).

Цель центрируется, поэтому свободный член = mean(y_train), признаки уже стандартизованы.
tau и m подбираются по контрольной части train; SVD считается один раз (слайд 13).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from common import get_regression_data, setup_style, save_fig, C1, C2, C3, MUTED


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


class SVDRegression:
    """alpha = U diag(w_j) V^T y, где веса w_j задают метод."""

    def fit(self, F, y):
        self.fmean_, self.ymean_ = F.mean(axis=0), y.mean()
        V, d, Ut = np.linalg.svd(F - self.fmean_, full_matrices=False)
        self.V_, self.d_, self.U_ = V, d, Ut.T
        self.Vty_ = V.T @ (y - self.ymean_)
        return self

    def alpha(self, kind="ols", tau=0.0, m=None):
        d = self.d_
        if kind == "ols":                       # 1/d_j  (слайд 8)
            w = 1.0 / d
        elif kind == "ridge":                   # d_j/(d_j^2+tau)  (слайд 12)
            w = d / (d ** 2 + tau)
        elif kind == "pcr":                     # 1/d_j при j<=m, иначе 0  (слайд 24)
            w = np.where(np.arange(len(d)) < m, 1.0 / d, 0.0)
        return self.U_ @ (w * self.Vty_)

    def predict(self, F, alpha):
        return (F - self.fmean_) @ alpha + self.ymean_


def main():
    setup_style()
    X_tr, X_te, y_tr, y_te, names = get_regression_data()
    n = X_tr.shape[1]

    # контрольная часть train для подбора tau и m (слайд 13)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)
    sel = SVDRegression().fit(X_fit, y_fit)
    taus = np.logspace(-2, 5, 50)
    q_tau = [rmse(y_val, sel.predict(X_val, sel.alpha("ridge", tau=t))) for t in taus]
    q_m = [rmse(y_val, sel.predict(X_val, sel.alpha("pcr", m=m))) for m in range(1, n + 1)]
    tau_best = float(taus[int(np.argmin(q_tau))])
    m_best = int(np.argmin(q_m)) + 1
    print(f"\nВыбор по контрольной части: tau* = {tau_best:.3g}, m* = {m_best}")
    print("RMSE_val(m):", np.round(q_m, 4))

    # итоговые модели на всём train
    full = SVDRegression().fit(X_tr, y_tr)
    lam = full.d_ ** 2
    models = {
        "МНК (SVD)": full.alpha("ols"),
        f"Ridge (SVD), tau={tau_best:.3g}": full.alpha("ridge", tau=tau_best),
        f"PCR (SVD), m={m_best}": full.alpha("pcr", m=m_best),
    }
    print("\nТест (RMSE, R^2):")
    own_pred = {}
    for name, a in models.items():
        p = full.predict(X_te, a)
        own_pred[name] = p
        print(f"  {name:<28} RMSE={rmse(y_te, p):.4f}  R2={r2(y_te, p):.4f}")

    # --- эталоны sklearn ---
    sk_ols = LinearRegression().fit(X_tr, y_tr)
    sk_ridge = Ridge(alpha=tau_best).fit(X_tr, y_tr)
    pca = PCA(n_components=m_best, svd_solver="full").fit(X_tr)
    sk_pcr = LinearRegression().fit(pca.transform(X_tr), y_tr)
    sk_pred = [sk_ols.predict(X_te), sk_ridge.predict(X_te),
               sk_pcr.predict(pca.transform(X_te))]
    print("\nmax |предсказание SVD - sklearn| на тесте:")
    for (name, p), q in zip(own_pred.items(), sk_pred):
        print(f"  {name:<28} {np.abs(p - q).max():.3e}")

    # --- слайды 9, 12, 14 ---
    a_ols = models["МНК (SVD)"]
    a_rdg = models[f"Ridge (SVD), tau={tau_best:.3g}"]
    print(f"\nmu(F^T F) = {lam[0] / lam[-1]:.2f}")
    print(f"||alpha_ols||^2 = {a_ols @ a_ols:.4f},  ||alpha_ridge||^2 = {a_rdg @ a_rdg:.4f}"
          f"  (слайд 14: ridge меньше)")
    print(f"эффективная размерность ridge: sum lambda/(lambda+tau) = "
          f"{np.sum(lam / (lam + tau_best)):.3f} < n = {n}")
    print("\nВеса (alpha):")
    print(f"  {'признак':<14}{'МНК':>10}{'Ridge':>10}{'PCR':>10}")
    a_pcr = models[f"PCR (SVD), m={m_best}"]
    for nm, a, b, c in zip(names, a_ols, a_rdg, a_pcr):
        print(f"  {nm:<14}{a:>10.4f}{b:>10.4f}{c:>10.4f}")

    # --- сжатие весов при росте tau (слайд 14) ---
    print("\nРидж при разных tau (train-SVD, метрики на тесте):")
    print(f"  {'tau':>8}{'||alpha||^2':>14}{'sum l/(l+tau)':>16}{'RMSE_test':>12}")
    for t in (0.01, 100, 1000, 5000, 20000):
        a = full.alpha("ridge", tau=t)
        print(f"  {t:>8g}{a @ a:>14.4f}{np.sum(lam / (lam + t)):>16.3f}"
              f"{rmse(y_te, full.predict(X_te, a)):>12.4f}")

    # --- PCR при разных m (слайд 24) и нагрузки компонент ---
    print("\nPCR при разных m (метрики на тесте):")
    for m in range(1, n + 1):
        p = full.predict(X_te, full.alpha("pcr", m=m))
        print(f"  m={m}  RMSE_test={rmse(y_te, p):.4f}  R2={r2(y_te, p):.4f}")
    print("\nНагрузки компонент (столбцы U, знак нормирован):")
    print(f"  {'признак':<14}" + "".join(f"{'ГК' + str(j + 1):>8}" for j in range(n)))
    for i, nm in enumerate(names):
        print(f"  {nm:<14}" + "".join(f"{full.U_[i, j]:>8.3f}" for j in range(n)))
    print("  корреляция scores ГК_j с целью (train):",
          np.round([np.corrcoef((X_tr - full.fmean_) @ full.U_[:, j], y_tr)[0, 1]
                    for j in range(n)], 3))

    # --- график: подбор tau и m ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.semilogx(taus, q_tau, "-", color=C1, label="RMSE на контроле")
    a1.plot(tau_best, min(q_tau), "o", color=C2, label=f"τ* = {tau_best:.3g}")
    a1.set_xlabel("τ (параметр регуляризации)")
    a1.set_ylabel("RMSE, Н·м")
    a1.set_title("Ridge: выбор τ (слайд 13)")
    a1.legend()
    a2.plot(range(1, n + 1), q_m, "o-", color=C1, label="RMSE на контроле")
    a2.plot(m_best, min(q_m), "s", color=C2, markersize=9, label=f"m* = {m_best}")
    a2.set_xlabel("m (число главных компонент)")
    a2.set_ylabel("RMSE, Н·м")
    a2.set_title("PCR: выбор m (слайд 24)")
    a2.legend()
    save_fig(fig, "regression_selection.png")


if __name__ == "__main__":
    main()
