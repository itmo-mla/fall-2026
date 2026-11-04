from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATHS = [ROOT / "weatherAUS.csv", ROOT.parent / "weatherAUS.csv", Path.cwd() / "weatherAUS.csv"]
OUTPUT_PATH = ROOT / "results.json"
SEED = 42
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SMOOTHING = 0.1
MARGIN_EPS = 0.15
MARGIN_REFRESH = 20
RANDOM_INIT_SCALE = 0.1
STEEPEST_ITERS = 200
N_STARTS = 10
N_SEEDS = 5
TAUS = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1]
MOMENTA = [0.0, 0.5, 0.9]
SGD_ETAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
MISSING_SHARE = 0.3
CLIP = 5.0
LOG_COLUMNS = ["Rainfall", "Evaporation"]
DELTAS = {
    "PressureDelta": ("Pressure3pm", "Pressure9am"),
    "HumidityDelta": ("Humidity3pm", "Humidity9am"),
    "TempRange": ("MaxTemp", "MinTemp"),
    "TempDelta": ("Temp3pm", "Temp9am"),
    "WindDelta": ("WindSpeed3pm", "WindSpeed9am"),
}
LOGREG_CS = [0.1, 1.0, 10.0]
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.85


def find_dataset() -> Path:
    for path in DATA_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Не найден weatherAUS.csv. Положи CSV рядом с linear_classifier.py.")


def clip_outliers(Z: np.ndarray) -> np.ndarray:
    return np.clip(Z, -CLIP, CLIP)


def prepare_dataset(path: Path) -> dict:
    raw = pd.read_csv(path)
    n_raw = len(raw)
    frame = raw.dropna(subset=["RainTomorrow"]).copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    if "RISK_MM" in frame.columns:
        frame = frame.drop(columns=["RISK_MM"])
    n = len(frame)
    unique_dates = frame["Date"].drop_duplicates().to_numpy()
    train_date_idx = max(1, int(len(unique_dates) * TRAIN_FRACTION))
    val_date_idx = max(train_date_idx + 1, int(len(unique_dates) * VAL_FRACTION))
    val_date_idx = min(val_date_idx, len(unique_dates) - 1)
    train_end = int(np.searchsorted(frame["Date"].to_numpy(), unique_dates[train_date_idx], side="left"))
    val_end = int(np.searchsorted(frame["Date"].to_numpy(), unique_dates[val_date_idx], side="left"))

    day = frame["Date"].dt.dayofyear
    frame["DaySin"] = np.sin(2.0 * np.pi * day / 365.25)
    frame["DayCos"] = np.cos(2.0 * np.pi * day / 365.25)
    date_start = str(frame["Date"].min().date())
    date_end = str(frame["Date"].max().date())
    for column in LOG_COLUMNS:
        frame[column] = np.log1p(frame[column])
    for name, (late, early) in DELTAS.items():
        frame[name] = frame[late] - frame[early]

    train_missing = frame.iloc[:train_end].isna().mean()
    sparse = [c for c in frame.columns if c not in ("Date", "RainTomorrow") and train_missing[c] > MISSING_SHARE]
    for column in sparse:
        frame[f"{column}Missing"] = frame[column].isna().astype(float)

    y = frame.pop("RainTomorrow").map({"No": -1, "Yes": 1}).to_numpy(dtype=int)
    frame = frame.drop(columns=["Date"])
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clip", FunctionTransformer(clip_outliers, feature_names_out="one-to-one")),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        verbose_feature_names_out=False,
    )
    transformer.fit(frame.iloc[:train_end])
    bounds = [(0, train_end), (train_end, val_end), (val_end, n)]
    X_parts = [transformer.transform(frame.iloc[a:b]).astype(np.float64) for a, b in bounds]
    names = transformer.get_feature_names_out().tolist() + ["bias"]
    X_parts = [np.column_stack([part, -np.ones(len(part))]) for part in X_parts]
    y_parts = [y[a:b] for a, b in bounds]

    num_block = X_parts[0][:, : len(numeric)]
    corr = np.corrcoef(num_block, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    upper = np.triu(abs_corr, k=1)
    pairs = np.argwhere(upper > 0.8)
    info = {
        "rows_raw": n_raw,
        "dropped_missing_target": n_raw - n,
        "rows_after_cleanup": n,
        "date_start": date_start,
        "date_end": date_end,
        "train_rows": len(y_parts[0]),
        "validation_rows": len(y_parts[1]),
        "test_rows": len(y_parts[2]),
        "features_after_encoding": len(names) - 1,
        "numeric_columns": len(numeric),
        "categorical_columns": len(categorical),
        "missing_indicator_columns": sparse,
        "log_columns": LOG_COLUMNS,
        "delta_features": list(DELTAS),
        "null_accuracy_test": float(max(np.mean(y_parts[2] == 1), np.mean(y_parts[2] == -1))),
        "clip": CLIP,
        "class_balance": {"No": int(np.sum(y == -1)), "Yes": int(np.sum(y == 1))},
        "train_class_balance": {"No": int(np.sum(y_parts[0] == -1)), "Yes": int(np.sum(y_parts[0] == 1))},
        "high_corr_pairs": [[numeric[i], numeric[j], float(corr[i, j])] for i, j in pairs],
        "mean_abs_corr": float(upper[np.triu_indices_from(upper, k=1)].mean()),
        "missing_share_train": {c: float(train_missing[c]) for c in train_missing.index if train_missing[c] > 0},
    }
    return {
        "X": X_parts,
        "y": y_parts,
        "names": names,
        "info": info,
        "numeric": numeric,
        "feature_corr": corr.tolist(),
    }


def margins(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return y * (X @ w)


def quadratic_loss(M: np.ndarray) -> np.ndarray:
    return (1.0 - M) ** 2


def gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> np.ndarray:
    M = margins(X, y, w)
    g = -(2.0 / len(y)) * (X.T @ ((1.0 - M) * y))
    g[:-1] += l2 * w[:-1]
    return g


def objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> float:
    return float(np.mean(quadratic_loss(margins(X, y, w))) + 0.5 * l2 * np.dot(w[:-1], w[:-1]))


def hessian(X: np.ndarray, l2: float) -> np.ndarray:
    H = (2.0 / len(X)) * (X.T @ X)
    H[np.arange(H.shape[0] - 1), np.arange(H.shape[0] - 1)] += l2
    return H


def steepest_step(X: np.ndarray, g: np.ndarray, l2: float) -> float:
    Xg = X @ g
    denominator = (2.0 / len(X)) * float(np.dot(Xg, Xg)) + l2 * float(np.dot(g[:-1], g[:-1]))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(g, g) / denominator)


def correlation_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for j in range(X.shape[1] - 1):
        denominator = float(np.dot(X[:, j], X[:, j]))
        if denominator > 1e-12:
            w[j] = float(np.dot(y, X[:, j])) / denominator
    return w


def random_init(n_weights: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-RANDOM_INIT_SCALE, RANDOM_INIT_SCALE, size=n_weights)


def margin_probabilities(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    p = 1.0 / (np.abs(margins(X, y, w)) + MARGIN_EPS)
    return p / p.sum()


def snapshot(history: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, smooth: float) -> None:
    M = margins(X, y, w)
    history["loss"].append(objective(X, y, w, l2))
    history["smooth_loss"].append(float(smooth))
    history["accuracy"].append(float(np.mean(M > 0)))
    history["mean_margin"].append(float(np.mean(M)))
    history["weight_norm"].append(float(np.linalg.norm(w[:-1])))


def new_history() -> dict:
    return {"loss": [], "smooth_loss": [], "accuracy": [], "mean_margin": [], "weight_norm": []}


def train_sgd(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    *,
    l2: float,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    momentum: float = MOMENTUM,
    sampling: str = "random",
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, dict]:
    if sampling not in {"random", "margin"}:
        raise ValueError("sampling must be 'random' or 'margin'")

    rng = np.random.default_rng(seed)
    n = len(y)
    steps = int(np.ceil(n / batch_size))
    w = init.astype(np.float64, copy=True)
    velocity = np.zeros_like(w)
    smooth = None
    history = new_history()
    snapshot(history, X, y, w, l2, objective(X, y, w, l2))

    for _ in range(epochs):
        order = rng.permutation(n) if sampling == "random" else None
        probabilities = None

        for step in range(steps):
            if sampling == "random":
                idx = order[step * batch_size : (step + 1) * batch_size]
            else:
                if step % MARGIN_REFRESH == 0:
                    probabilities = margin_probabilities(X, y, w)
                idx = rng.choice(n, size=batch_size, replace=True, p=probabilities)

            xb, yb = X[idx], y[idx]
            g = gradient(xb, yb, w, l2)
            xi = objective(xb, yb, w, l2)
            smooth = xi if smooth is None else (1.0 - SMOOTHING) * smooth + SMOOTHING * xi

            velocity = momentum * velocity + lr * g
            w = w - velocity

        snapshot(history, X, y, w, l2, smooth if smooth is not None else objective(X, y, w, l2))

    return w, history


def train_steepest(X: np.ndarray, y: np.ndarray, init: np.ndarray, *, l2: float, iters: int = STEEPEST_ITERS) -> tuple[np.ndarray, dict]:
    w = init.astype(np.float64, copy=True)
    smooth = objective(X, y, w, l2)
    history = new_history()
    history["step_size"] = []
    snapshot(history, X, y, w, l2, smooth)
    for _ in range(iters):
        g = gradient(X, y, w, l2)
        h = steepest_step(X, g, l2)
        w = w - h * g
        xi = objective(X, y, w, l2)
        smooth = (1.0 - SMOOTHING) * smooth + SMOOTHING * xi
        snapshot(history, X, y, w, l2, smooth)
        history["step_size"].append(h)
    return w, history


def check_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, eps: float = 1e-6) -> float:
    analytic = gradient(X, y, w, l2)
    numeric = np.zeros_like(w, dtype=np.float64)
    for j in range(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[j] += eps
        wm[j] -= eps
        numeric[j] = (objective(X, y, wp, l2) - objective(X, y, wm, l2)) / (2.0 * eps)
    return float(np.max(np.abs(analytic - numeric)))


def closed_form(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    A = hessian(X, l2)
    b = (2.0 / len(y)) * (X.T @ y.astype(np.float64))
    return np.linalg.lstsq(A, b, rcond=None)[0]


def sklearn_reference(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    best_w, best_score, best_eta = None, -np.inf, None
    for eta in SGD_ETAS:
        model = SGDClassifier(
            loss="squared_error",
            penalty="l2",
            alpha=l2 / 2.0,
            max_iter=EPOCHS,
            tol=None,
            learning_rate="constant",
            eta0=eta,
            random_state=SEED,
        )
        model.fit(X[:, :-1], y)
        w = np.concatenate([model.coef_[0], [-model.intercept_[0]]])
        score = average_precision_score((y_val == 1).astype(int), X_val @ w)
        if score > best_score:
            best_w, best_score, best_eta = w, score, eta
    return best_w, best_eta


def logistic_reference(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, float]:
    best_w, best_score, best_c = None, -np.inf, None
    for c in LOGREG_CS:
        model = LogisticRegression(C=c, max_iter=500).fit(X[:, :-1], y)
        w = np.concatenate([model.coef_[0], [-model.intercept_[0]]])
        score = average_precision_score((y_val == 1).astype(int), X_val @ w)
        if score > best_score:
            best_w, best_score, best_c = w, score, c
    return best_w, best_c


def margin_stats(M: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(M)),
        "median": float(np.median(M)),
        "min": float(np.min(M)),
        "max": float(np.max(M)),
        "q05": float(np.quantile(M, 0.05)),
        "q95": float(np.quantile(M, 0.95)),
        "share_negative": float(np.mean(M < 0)),
        "share_near_boundary": float(np.mean(np.abs(M) < MARGIN_EPS)),
        "share_above_one": float(np.mean(M >= 1.0)),
    }


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve((y == 1).astype(int), scores)
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, threshold: float = 0.0) -> dict:
    scores = X @ w
    pred = np.where(scores >= threshold, 1, -1)
    y01 = (y == 1).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y01, scores)),
        "loss": objective(X, y, w, l2),
        "weight_norm": float(np.linalg.norm(w[:-1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[-1, 1]).tolist(),
    }


def summarize(data: dict, w: np.ndarray, l2: float, history: dict | None = None) -> dict:
    X_train, X_val, X_test = data["X"]
    y_train, y_val, y_test = data["y"]
    threshold = best_threshold(X_val @ w, y_val)
    entry = {
        "validation": evaluate(X_val, y_val, w, l2),
        "test": evaluate(X_test, y_test, w, l2),
        "tuned_threshold": threshold,
        "validation_tuned_threshold": evaluate(X_val, y_val, w, l2, threshold),
        "test_tuned_threshold": evaluate(X_test, y_test, w, l2, threshold),
        "train_loss": objective(X_train, y_train, w, l2),
    }
    if history is not None:
        entry["history"] = history
    return entry


def mean_std(runs: list[dict], keys: list[str]) -> dict:
    return {
        k: {"mean": float(np.mean([r["test"][k] for r in runs])), "std": float(np.std([r["test"][k] for r in runs]))}
        for k in keys
    }


def main() -> None:
    started = time.perf_counter()
    path = find_dataset()
    data = prepare_dataset(path)
    X_train, X_val, X_test = data["X"]
    y_train, y_val, y_test = data["y"]
    info = data["info"]
    names = data["names"]
    n_weights = X_train.shape[1]

    w_corr = correlation_init(X_train, y_train)
    w_rand = random_init(n_weights, np.random.default_rng(SEED))

    tau_rows = []
    for tau in TAUS:
        w, _ = train_sgd(X_train, y_train, w_corr, l2=tau)
        row = summarize(data, w, tau)
        w_cf = closed_form(X_train, y_train, tau)
        row_cf = summarize(data, w_cf, tau)
        tau_rows.append(
            {
                "tau": tau,
                "validation": row["validation"],
                "test": row["test"],
                "closed_form_validation": row_cf["validation"],
                "closed_form_test": row_cf["test"],
            }
        )
    l2 = max((r for r in tau_rows if r["tau"] > 0), key=lambda r: r["closed_form_validation"]["pr_auc"])["tau"]

    w_star = closed_form(X_train, y_train, l2)
    q_star = objective(X_train, y_train, w_star, l2)
    optimum = summarize(data, w_star, l2)
    optimum["train_loss"] = q_star

    inertia = {}
    for mu in MOMENTA:
        w, hist = train_sgd(X_train, y_train, w_corr, l2=l2, momentum=mu)
        inertia[f"{mu}"] = summarize(data, w, l2, hist)
        inertia[f"{mu}"]["momentum"] = mu
        inertia[f"{mu}"]["weights"] = w

    w_corr_sgd = inertia[f"{MOMENTUM}"]["weights"]
    experiments = {"correlation_init": {k: v for k, v in inertia[f"{MOMENTUM}"].items() if k != "weights"}}
    experiments["correlation_init"]["initial_accuracy"] = float(np.mean(margins(X_train, y_train, w_corr) > 0))

    w_margin, hist_margin = train_sgd(X_train, y_train, w_corr, l2=l2, sampling="margin")
    experiments["margin_sampling"] = summarize(data, w_margin, l2, hist_margin)

    runs = []
    run_weights = []
    rng = np.random.default_rng(SEED)
    for k in range(N_STARTS):
        w, hist = train_sgd(X_train, y_train, random_init(n_weights, rng), l2=l2, seed=SEED + k + 1)
        entry = summarize(data, w, l2, hist)
        entry["run"] = k + 1
        runs.append(entry)
        run_weights.append(w)
    best_run = max(runs, key=lambda r: r["validation"]["pr_auc"])
    experiments["multistart"] = {
        "runs": runs,
        "best_run": best_run["run"],
        "validation_pr_auc_mean": float(np.mean([r["validation"]["pr_auc"] for r in runs])),
        "validation_pr_auc_std": float(np.std([r["validation"]["pr_auc"] for r in runs])),
        "test_pr_auc_mean": float(np.mean([r["test"]["pr_auc"] for r in runs])),
        "test_pr_auc_std": float(np.std([r["test"]["pr_auc"] for r in runs])),
        "final_gap_mean": float(np.mean([r["history"]["loss"][-1] - q_star for r in runs])),
        "final_gap_std": float(np.std([r["history"]["loss"][-1] - q_star for r in runs])),
    }
    w_multi = run_weights[best_run["run"] - 1]

    w_margin_rand, hist_margin_rand = train_sgd(X_train, y_train, w_rand, l2=l2, sampling="margin", seed=SEED + 10)
    experiments["margin_sampling_random_init"] = summarize(data, w_margin_rand, l2, hist_margin_rand)

    w_fast, hist_fast = train_steepest(X_train, y_train, w_corr, l2=l2)
    experiments["fastest_descent"] = summarize(data, w_fast, l2, hist_fast)

    seed_study = {}
    for sampling in ["random", "margin"]:
        study = []
        for s in range(N_SEEDS):
            w, _ = train_sgd(X_train, y_train, w_corr, l2=l2, sampling=sampling, seed=SEED + 100 + s)
            study.append(summarize(data, w, l2))
        seed_study[sampling] = mean_std(study, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "loss"])
    experiments["seed_study"] = seed_study

    ref_w, ref_eta = sklearn_reference(X_train, y_train, X_val, y_val, l2)
    reference = summarize(data, ref_w, l2)
    reference["eta0"] = ref_eta

    logreg_w, logreg_c = logistic_reference(X_train, y_train, X_val, y_val)
    logistic = summarize(data, logreg_w, l2)
    logistic["C"] = logreg_c

    candidates = {
        "correlation_init": w_corr_sgd,
        "margin_sampling": w_margin,
        "margin_sampling_random_init": w_margin_rand,
        "multistart": w_multi,
        "fastest_descent": w_fast,
    }
    chosen_name = max(candidates, key=lambda n: evaluate(X_val, y_val, candidates[n], l2)["pr_auc"])
    chosen_w = candidates[chosen_name]
    threshold = best_threshold(X_val @ chosen_w, y_val)
    chosen_histories = {
        "correlation_init": experiments["correlation_init"]["history"],
        "margin_sampling": hist_margin,
        "margin_sampling_random_init": hist_margin_rand,
        "multistart": best_run["history"],
        "fastest_descent": hist_fast,
    }
    test_scores = X_test @ chosen_w
    test_prob = np.clip((1.0 + test_scores) / 2.0, 0.0, 1.0)
    frac_pos, mean_pred = calibration_curve(
        (y_test == 1).astype(int), test_prob, n_bins=10, strategy="quantile"
    )
    best_test_metrics = evaluate(X_test, y_test, chosen_w, l2)
    best_test_metrics["brier"] = float(brier_score_loss((y_test == 1).astype(int), test_prob))
    best_test_tuned_metrics = evaluate(X_test, y_test, chosen_w, l2, threshold)
    best_test_tuned_metrics["brier"] = best_test_metrics["brier"]
    top = np.argsort(-np.abs(chosen_w[:-1]))[:15]
    gradient_error = check_gradient(
        X_train[: min(256, len(X_train))],
        y_train[: min(256, len(y_train))],
        np.random.default_rng(SEED).normal(0, 0.1, size=X_train.shape[1]),
        l2,
    )

    M_test_best = margins(X_test, y_test, chosen_w)
    grid = np.linspace(0.0, 1.0, 501)
    results = {
        "dataset_path": path.name,
        "settings": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "smoothing_lambda": SMOOTHING,
            "margin_eps": MARGIN_EPS,
            "margin_refresh_steps": MARGIN_REFRESH,
            "random_init_scale": RANDOM_INIT_SCALE,
            "l2_chosen_by_closed_form_validation_pr_auc": l2,
            "multistart_runs": N_STARTS,
            "seed_study_runs": N_SEEDS,
            "steepest_iterations": STEEPEST_ITERS,
        },
        "split": info,
        "tau_sweep": tau_rows,
        "optimum_closed_form": optimum,
        "inertia": {k: {kk: vv for kk, vv in v.items() if kk != "weights"} for k, v in inertia.items()},
        "experiments": experiments,
        "reference_sgd_classifier": reference,
        "reference_logistic_regression": logistic,
        "best_model": {
            "name": chosen_name,
            "selected_by": "validation pr_auc",
            "validation": evaluate(X_val, y_val, chosen_w, l2),
            "test": best_test_metrics,
            "validation_tuned_threshold": evaluate(X_val, y_val, chosen_w, l2, threshold),
            "test_tuned_threshold": best_test_tuned_metrics,
            "history": chosen_histories[chosen_name],
            "top_weights": {names[i]: float(chosen_w[i]) for i in top},
            "calibration": {
                "method": "p = clip((1 + f) / 2, 0, 1)",
                "mean_predicted": mean_pred.tolist(),
                "fraction_positive": frac_pos.tolist(),
            },
        },
        "margin_analysis": {
            "train_correlation_init": margin_stats(margins(X_train, y_train, w_corr)),
            "train_random_init": margin_stats(margins(X_train, y_train, w_rand)),
            "train_best": margin_stats(margins(X_train, y_train, chosen_w)),
            "test_best": margin_stats(M_test_best),
            "test_best_class_yes": margin_stats(M_test_best[y_test == 1]),
            "test_best_class_no": margin_stats(M_test_best[y_test == -1]),
            "quantile_grid": grid.tolist(),
            "train_correlation_init_quantiles": np.quantile(margins(X_train, y_train, w_corr), grid).tolist(),
            "train_random_init_quantiles": np.quantile(margins(X_train, y_train, w_rand), grid).tolist(),
            "train_best_quantiles": np.quantile(margins(X_train, y_train, chosen_w), grid).tolist(),
        },
        "gradient_check_max_abs_error": gradient_error,
        "runtime_seconds": float(time.perf_counter() - started),
        "test_margins": M_test_best.tolist(),
        "test_labels": y_test.tolist(),
        "test_scores": test_scores.tolist(),
    }
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()