
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
RANDOM_STATE = 42
TEST_SIZE = 0.30
C = 1.0
C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
TWO_FEATURE_TRAIN_LIMIT = 190

FEATURE_COLUMNS = (0, 1)
FEATURE_LABELS = ("Средний радиус (стандартизирован)", "Средняя текстура (стандартизирована)")
SHOW_PLOTS = True
PLOTS_DIR = Path("plots")


COLORS = {-1: "#D95459", 1: "#3268BF"}
CLASS_NAMES = {-1: "B — доброкачественное (−1)", 1: "M — злокачественное (+1)"}



def load_data(source=DATA_URL):
    try:
        df = pd.read_csv(source, header=None)
        if df.shape != (569, 32):
            raise ValueError(f"Ожидалась исходная таблица WDBC (569, 32), получено {df.shape}")
        X = df.iloc[:, 2:].to_numpy(dtype=float)
        diagnosis = df.iloc[:, 1]
        if not diagnosis.isin(["B", "M"]).all():
            raise ValueError("Неожиданные метки: ожидаются B и M")
        y = diagnosis.map({"B": -1, "M": 1}).to_numpy(dtype=int)
        print("Источник данных: UCI WDBC (wdbc.data)")
    except (OSError, ConnectionError) as exc:
        ds = load_breast_cancer()
        X = ds.data.astype(float)
        y = np.where(ds.target == 0, 1, -1).astype(int)
        print(f"UCI недоступен ({type(exc).__name__}); используем встроенную копию WDBC sklearn")
    if not np.isfinite(X).all() or not np.isin(y, [-1, 1]).all():
        raise ValueError("Данные повреждены или содержат пропуски")
    return X, y


def prepare_data(X, y, test_size=TEST_SIZE):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te), y_tr, y_te


def gamma_scale(X):
    return 1.0 / (X.shape[1] * X.var())


def metric_values(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_M": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_M": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_M": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def kernel_matrix(A, B, kind="linear", gamma=None):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if kind == "linear":
        return A @ B.T
    if kind == "rbf":
        if gamma is None or gamma <= 0:
            raise ValueError("Для RBF нужно задать положительное gamma")
        aa = np.sum(A ** 2, axis=1)[:, None]
        bb = np.sum(B ** 2, axis=1)[None, :]
        squared_distances = np.maximum(aa + bb - 2 * A @ B.T, 0.0)
        return np.exp(-gamma * squared_distances)
    raise ValueError("Поддерживаются ядра 'linear' и 'rbf'")



def dual_objective(lambdas, Q):
    return 0.5 * lambdas @ Q @ lambdas - np.sum(lambdas)


def dual_gradient(lambdas, Q):
    return Q @ lambdas - np.ones_like(lambdas)


def equality_constraint(lambdas, y):
    return lambdas @ y


def equality_gradient(lambdas, y):
    return y


def solve_dual(X, y, C=1.0, kind="linear", gamma=None):
    if C <= 0:
        raise ValueError("Параметр C должен быть положительным")
    K = kernel_matrix(X, X, kind=kind, gamma=gamma)
    Q = np.outer(y, y) * K
    constraint = {
        "type": "eq",
        "fun": equality_constraint,
        "jac": equality_gradient,
        "args": (y,),
    }
    result = minimize(
        fun=dual_objective,
        x0=np.zeros(len(y)),
        args=(Q,),
        jac=dual_gradient,
        method="SLSQP",
        bounds=[(0.0, C)] * len(y),
        constraints=[constraint],
        options={"maxiter": 2200, "ftol": 1e-9, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP не сошёлся: {result.message}")
    return result.x, K

def compute_intercept(lambdas, y, K, C, tol):
    free = np.flatnonzero((lambdas > tol) & (lambdas < C - tol))
    scores_without_b = K @ (lambdas * y)
    if len(free):
        return float(np.mean(y[free] - scores_without_b[free]))

    lower, upper = [], []
    for a, yi, score in zip(lambdas, y, scores_without_b):
        margin_b = yi - score
        if a <= tol:
            (lower if yi > 0 else upper).append(margin_b)
        elif a >= C - tol:
            (upper if yi > 0 else lower).append(margin_b)
    if not lower or not upper:
        raise RuntimeError("Не удалось восстановить b из условий KKT")
    return float((max(lower) + min(upper)) / 2)


def fit_svm(X, y, C=1.0, kind="linear", gamma=None, tol=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isin(y, [-1, 1]).all():
        raise ValueError("Ожидаются X размера (n,d) и метки y из {-1,+1}")
    if len(np.unique(y)) < 2:
        raise ValueError("В обучении должны присутствовать оба класса")
    if C <= 0:
        raise ValueError("Параметр C должен быть положительным")
    if tol is None:
        tol = min(1e-6, C * 1e-3)
    if not 0 < tol < C / 2:
        raise ValueError("tol должен быть положительным и меньше C/2")

    lambdas, K = solve_dual(X, y, C=C, kind=kind, gamma=gamma)
    support_indices = np.flatnonzero(lambdas > tol)
    if not len(support_indices):
        raise RuntimeError("Оптимизатор не определил опорные векторы")
    b = compute_intercept(lambdas, y, K, C=C, tol=tol)
    w = (lambdas * y) @ X if kind == "linear" else None
    return {
        "kind": kind,
        "gamma": gamma,
        "C": C,
        "lambdas": lambdas,
        "support_indices": support_indices,
        "support_vectors": X[support_indices],
        "dual_coef": lambdas[support_indices] * y[support_indices],
        "w": w,
        "b": b,
    }

def decision_function(model, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if model["kind"] == "linear":
        return X @ model["w"] + model["b"]
    K = kernel_matrix(model["support_vectors"], X,
                      kind=model["kind"], gamma=model["gamma"])
    return model["dual_coef"] @ K + model["b"]


def predict(model, X):
    return np.where(decision_function(model, X) >= 0, 1, -1)



def get_scores(model, X):
    return decision_function(model, X) if isinstance(model, dict) else model.decision_function(X)


def get_sv(model):
    return model["support_vectors"] if isinstance(model, dict) else model.support_vectors_


def get_pred(model, X):
    return predict(model, X) if isinstance(model, dict) else model.predict(X)


def n_sv(model):
    return len(get_sv(model))


def save_show(fig, filename, show=True):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOTS_DIR / filename
    fig.savefig(output, dpi=165, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    print("Сохранено:", output)
    return output


def scatter_classes(ax, X, y, *, point_size=31):
    for cls in (-1, 1):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=point_size,
                   c=COLORS[cls], edgecolors="white", linewidths=0.5,
                   alpha=0.82, label=CLASS_NAMES[cls], zorder=4)


def legend_handles(include_sv=True):
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS[-1],
               markeredgecolor="white", markersize=7, label=CLASS_NAMES[-1]),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS[1],
               markeredgecolor="white", markersize=7, label=CLASS_NAMES[1]),
        Patch(facecolor="#f7dfe0", edgecolor="none", label="Область прогноза B"),
        Patch(facecolor="#dce7fa", edgecolor="none", label="Область прогноза M"),
        Line2D([], [], c="#202020", linewidth=1.6, label="Граница f(x)=0"),
        Line2D([], [], c="#606874", linestyle="--", linewidth=1.1, label="Отступ f(x)=±1"),
    ]
    if include_sv:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=8,
                              markerfacecolor="none", markeredgecolor="black",
                              label="Опорные векторы"))
    return handles


def make_grid(X, step_count=175):
    dx, dy = np.ptp(X, axis=0)
    padx, pady = max(.65, dx * .09), max(.65, dy * .09)
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-padx, X[:, 0].max()+padx, step_count),
        np.linspace(X[:, 1].min()-pady, X[:, 1].max()+pady, step_count),
    )
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def plot_boundary_on_ax(ax, model, X, y, title, show_sv=True, grid=None, legend=True):
    if grid is None:
        grid = make_grid(X)
    xx, yy, points = grid
    values = get_scores(model, points).reshape(xx.shape)
    ax.contourf(xx, yy, values >= 0, levels=[-.5, .5, 1.5],
                colors=["#f7dfe0", "#dce7fa"], alpha=.78)
    if values.min() < 0 < values.max():
        ax.contour(xx, yy, values, levels=[0], colors=["#202020"], linewidths=1.6)
    levels = [level for level in (-1, 1) if values.min() < level < values.max()]
    if levels:
        ax.contour(xx, yy, values, levels=levels, colors=["#606874"],
                   linestyles="--", linewidths=1.1)
    scatter_classes(ax, X, y, point_size=27)
    if show_sv:
        sv = get_sv(model)
        ax.scatter(sv[:, 0], sv[:, 1], s=77, facecolors="none",
                   edgecolors="black", linewidths=1.1, zorder=6)
    ax.set_title(title, fontsize=11.5, fontweight="semibold", pad=11)
    ax.set_xlabel(FEATURE_LABELS[0])
    ax.set_ylabel(FEATURE_LABELS[1])
    ax.grid(True, linestyle="--", alpha=.15)
    ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    if legend:
        ax.legend(handles=legend_handles(include_sv=show_sv), loc="upper right",
                  fontsize=8.2, framealpha=.96, facecolor="white")


def plot_original_features(X_2d, y, show=True):
    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
    scatter_classes(ax, X_2d, y, point_size=32)
    ax.set_title("01. WDBC: средний радиус и средняя текстура", fontsize=13, weight="bold")
    ax.set_xlabel(FEATURE_LABELS[0]); ax.set_ylabel(FEATURE_LABELS[1])
    ax.legend(loc="best", framealpha=.96)
    ax.grid(True, linestyle="--", alpha=.25)
    ax.text(.02, .02, "Два исходных признака WDBC, выбранных напрямую, без PCA.",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    return save_show(fig, "01_original_features.png", show)


def plot_linear_single(model, X, y, train_acc, test_acc, show=True):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(left=.08, right=.72, bottom=.10, top=.90)
    plot_boundary_on_ax(ax, model, X, y,
        f"02. Линейный SVM: mean radius / mean texture\nAccuracy: train={train_acc:.3f}, test={test_acc:.3f}",
        legend=False)
    ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=10, framealpha=.98)
    return save_show(fig, "02_linear_boundary.png", show)


def plot_kernel_comparison(model_linear, model_rbf, X, y, metrics_linear, metrics_rbf, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.99, top=.83, bottom=.21, wspace=.14)
    grid = make_grid(X)
    for ax, model, k, m in zip(axes, [model_linear, model_rbf],
                               ["Линейное ядро", "RBF-ядро"],
                               [metrics_linear, metrics_rbf]):
        plot_boundary_on_ax(ax, model, X, y,
            f"{k} (C=1)\nAccuracy: train={m['train']:.3f}, test={m['test']:.3f}",
            grid=grid, legend=False)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(.52, .015),
               ncol=4, fontsize=9.5, framealpha=.97)
    fig.suptitle("03. Сравнение ядер — WDBC, два исходных признака",
                 fontsize=15, weight="bold", y=.975)
    return save_show(fig, "03_linear_vs_rbf.png", show)

def plot_c_effect(results, show=True):
    c = np.array([r["C"] for r in results]); tr = [r["train"] for r in results]
    te = [r["test"] for r in results]; sv = [r["sv"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.7), constrained_layout=True)
    ax1.semilogx(c, tr, "o-", linewidth=2, c="#3268BF", label="Train accuracy")
    ax1.semilogx(c, te, "s-", linewidth=2, c="#D95459", label="Test accuracy")
    for x, v in zip(c, te):
        ax1.annotate(f"{v:.3f}", (x, v), xytext=(0, -15), textcoords="offset points",
                     fontsize=8, ha="center")
    ax1.set_xlabel("C (логарифмическая шкала)"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Влияние C на точность RBF SVM")
    ax1.set_ylim(max(0, min(tr+te)-.09), min(1.015, max(tr+te)+.06))
    ax1.legend(loc="best", framealpha=.96); ax1.grid(True, alpha=.25)
    ax2.semilogx(c, sv, "D-", c="#238267", linewidth=2, label="Число опорных векторов")
    for x, v in zip(c, sv):
        ax2.annotate(str(v), (x, v), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9)
    ax2.set_xlabel("C (логарифмическая шкала)"); ax2.set_ylabel("Число опорных векторов")
    ax2.set_title("Влияние C на число опорных векторов")
    ax2.legend(loc="best", framealpha=.96); ax2.grid(True, alpha=.25)
    fig.suptitle("04. Влияние C (RBF на двух исходных признаках)",
                 fontsize=14, weight="bold")
    return save_show(fig, "04_c_metrics.png", show)



def plot_c_boundaries(results, X, y, show=True):
    fig, axes = plt.subplots(2, 3, figsize=(18.2, 11), constrained_layout=True,
                             sharex=True, sharey=True)
    grid = make_grid(X)
    for ax, r in zip(axes.flat, results):
        plot_boundary_on_ax(ax, r["model"], X, y,
            f"RBF, C={r['C']:g}\nTrain={r['train']:.3f}; test={r['test']:.3f}",
            show_sv=True, grid=grid, legend=False)
        if len(np.unique(predict(r["model"], grid[2]))) == 1:
            ax.text(.02,.025,"В видимой области нет границы f(x)=0",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white",alpha=.92,edgecolor="none"))
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=legend_handles(), loc="center left", fontsize=12,
                         frameon=False, title="Обозначения", title_fontsize=14)
    fig.suptitle("05. Как C меняет границу RBF (mean radius / mean texture)",
                 fontsize=15, weight="bold")
    return save_show(fig, "05_c_boundaries.png", show)


def comparison_dataframe(rows):
    table = pd.DataFrame([
        {
            "Модель": r["name"],
            "Train Acc": r["train_accuracy"],
            "Test Acc": r["accuracy"],
            "Precision M": r["precision_M"],
            "Recall M": r["recall_M"],
            "F1 M": r["f1_M"],
            "SV": r["sv"],
        }
        for r in rows
    ])
    return table.set_index("Модель")


def print_comparison_table(rows):
    table = comparison_dataframe(rows)
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Положительный класс M (+1); SV — количество опорных векторов.")
    return table



def plot_confusions(y_test, models, X_test, title, filename, show=True):
    cms = [confusion_matrix(y_test, get_pred(m, X_test), labels=[-1, 1])
           for _,m in models]
    vmax = max(1, max(cm.max() for cm in cms))
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 10.4))
    fig.suptitle(title, fontsize=15, weight="bold", y=.975)
    fig.subplots_adjust(left=.09, right=.85, top=.92, bottom=.15, hspace=.41, wspace=.27)
    codes = np.array([["TN", "FP"], ["FN", "TP"]])
    for ax, (name, model), cm in zip(axes.flat, models, cms):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks([0,1], ["B (−1)", "M (+1)"])
        ax.set_yticks([0,1], ["B (−1)", "M (+1)"])
        ax.set_xlabel("Предсказанный класс"); ax.set_ylabel("Истинный класс")
        ax.set_title(name, fontsize=11.5, weight="semibold")
        ax.tick_params(length=0)
        for i in range(2):
            for j in range(2):
                number = int(cm[i,j]); right = i == j
                ax.add_patch(Rectangle((j-.5, i-.5),1,1, fill=False, lw=2.5,
                            ec="#198567" if right else "#d35454"))
                ax.text(j, i, f"{codes[i,j]}\n{number}", ha="center", va="center",
                        fontsize=14, weight="bold", color="white" if number > .48*vmax else "#25313e")
        ax.set_xlim(-.52,1.52); ax.set_ylim(1.52,-.52)
    cax = fig.add_axes([.884, .23, .018, .54]); cb = fig.colorbar(im,cax=cax)
    cb.set_label("Количество объектов")
    fig.legend(handles=[Patch(facecolor="none", ec="#198567", linewidth=2.5, label="Верно: TN (B→B), TP (M→M)"),
                        Patch(facecolor="none", ec="#d35454", linewidth=2.5, label="Ошибка: FP (B→M), FN (M→B)")],
               loc="lower center", bbox_to_anchor=(.48,.055), ncol=2, frameon=True, fontsize=10)
    fig.text(.48,.025,"Особенно важны FN: злокачественное образование ошибочно определено как доброкачественное.",
             ha="center", fontsize=9.5)
    return save_show(fig, filename, show)


def main(data_source=DATA_URL, show=SHOW_PLOTS, plots_dir=PLOTS_DIR,
         two_feature_train_limit=TWO_FEATURE_TRAIN_LIMIT, c_values=C_VALUES):
    global PLOTS_DIR
    PLOTS_DIR = Path(plots_dir)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_source)
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    print(f"Размерность WDBC: {X.shape}; train={X_train.shape}, test={X_test.shape}")
    print(f"Классы: B=-1 ({sum(y==-1)}), M=+1 ({sum(y==1)})")

    gamma_full = gamma_scale(X_train)
    models_full = [
        ("Свой Linear", fit_svm(X_train,y_train,C=C,kind="linear")),
        ("sklearn Linear", SVC(C=C,kernel="linear",tol=1e-6).fit(X_train,y_train)),
        ("Свой RBF", fit_svm(X_train,y_train,C=C,kind="rbf",gamma=gamma_full)),
        ("sklearn RBF", SVC(C=C,kernel="rbf",gamma=gamma_full,tol=1e-6).fit(X_train,y_train)),
    ]
    rows=[]
    for name,model in models_full:
        m=metric_values(y_test,get_pred(model,X_test))
        rows.append(dict(name=name, train_accuracy=accuracy_score(y_train,get_pred(model,X_train)),
                         sv=n_sv(model), **m))
    print_comparison_table(rows)
    print("Совпадение Linear (свой / sklearn):", np.mean(get_pred(models_full[0][1],X_test)==get_pred(models_full[1][1],X_test)))
    print("Совпадение RBF (свой / sklearn):", np.mean(get_pred(models_full[2][1],X_test)==get_pred(models_full[3][1],X_test)))
    print("Проверка двойственного равенства Linear:",models_full[0][1]["lambdas"]@y_train)
    print("Проверка двойственного равенства RBF:",models_full[2][1]["lambdas"]@y_train)

    X_train_2d_all = X_train[:, FEATURE_COLUMNS]
    X_test_2d = X_test[:, FEATURE_COLUMNS]
    if two_feature_train_limit is not None and two_feature_train_limit < len(y_train):
        X_2d, _, y_2d, _ = train_test_split(
            X_train_2d_all, y_train, train_size=two_feature_train_limit,
            stratify=y_train, random_state=RANDOM_STATE)
    else:
        X_2d, y_2d = X_train_2d_all, y_train
    print(f"Для двумерных графиков и C: {len(y_2d)} train, {len(y_test)} test; два исходных признака: mean radius, mean texture")
    gamma_2d = gamma_scale(X_2d)

    plot_original_features(X_2d, y_2d, show)

    linear_2d = fit_svm(X_2d, y_2d, C=C, kind="linear")
    linear_info = {"train": accuracy_score(y_2d, predict(linear_2d, X_2d)),
                   "test": accuracy_score(y_test, predict(linear_2d, X_test_2d))}
    plot_linear_single(linear_2d, X_2d, y_2d, linear_info["train"], linear_info["test"], show)

    sweep=[]
    for c in c_values:
        model=fit_svm(X_2d, y_2d, C=c, kind="rbf", gamma=gamma_2d)
        row={"C":float(c), "model":model,
             "train":accuracy_score(y_2d,predict(model,X_2d)),
             "test":accuracy_score(y_test,predict(model,X_test_2d)),
             "sv":n_sv(model)}
        sweep.append(row)
        print(f"2 features RBF C={c:7g}: train={row['train']:.4f}, test={row['test']:.4f}, SV={row['sv']}")
    base=next((r for r in sweep if np.isclose(r['C'],C)),None)
    if base is None:
        base_model=fit_svm(X_2d,y_2d,C=C,kind="rbf",gamma=gamma_2d)
        base={"model":base_model,"train":accuracy_score(y_2d,predict(base_model,X_2d)),
              "test":accuracy_score(y_test,predict(base_model,X_test_2d))}
    plot_kernel_comparison(linear_2d,base["model"],X_2d,y_2d,linear_info,base,show)
    plot_c_effect(sweep,show)
    plot_c_boundaries(sweep,X_2d,y_2d,show)
    plot_confusions(y_test,models_full,X_test,
                    "07. Матрицы ошибок: все 30 признаков WDBC", "07_confusion_full.png", show)
    models_2d=[("Свой Linear (2 признака)",linear_2d),
               ("sklearn Linear (2 признака)",SVC(C=C,kernel="linear",tol=1e-6).fit(X_2d,y_2d)),
               ("Свой RBF (2 признака)",base["model"]),
               ("sklearn RBF (2 признака)",SVC(C=C,kernel="rbf",gamma=gamma_2d,tol=1e-6).fit(X_2d,y_2d))]
    plot_confusions(y_test,models_2d,X_test_2d,
                    "08. Матрицы ошибок: mean radius и mean texture", "08_confusion_two_features.png", show)
    return {"full_models":models_full,"full_metrics":rows,"two_feature_models":models_2d,
            "two_feature_data":(X_2d,y_2d,X_test_2d,y_test),"c_results":sweep}

if __name__ == "__main__":
    main()
