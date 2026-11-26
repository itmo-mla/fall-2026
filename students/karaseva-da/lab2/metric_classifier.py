from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATHS = [ROOT / "Penguin Species Prediction Dataset.csv", Path.cwd() / "Penguin Species Prediction Dataset.csv"]
OUTPUT_PATH = ROOT / "results.json"
FIGURE_PATH = ROOT / "figures"
K_VALUES = list(range(1, 101))
RETAINED_FRACTIONS = [1.0, 0.9, 0.75, 0.5, 0.35, 0.25, 0.2, 0.15, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
MAX_RISK_INCREASE = 0.005
INNER_FOLDS = 3
SEED = 42
BANDWIDTHS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
NUMERIC_COLUMNS = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_COLUMNS = ["island", "sex"]


def find_dataset() -> Path:
    for path in DATA_PATHS:
        if path.is_file():
            return path
    import kagglehub
    dataset_dir = Path(kagglehub.dataset_download("srisyra02/penguin-species-dataset"))
    return next(dataset_dir.rglob("Penguin Species Prediction Dataset.csv"))


def read_dataset(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    classes, y = np.unique([row["species"] for row in rows], return_inverse=True)
    numeric = np.array([[float(row[c]) if row[c].strip() else np.nan for c in NUMERIC_COLUMNS] for row in rows])
    categorical = np.array([[row[c].strip() or "missing" for c in CATEGORICAL_COLUMNS] for row in rows])
    missing = {c: int(np.isnan(numeric[:, j]).sum()) for j, c in enumerate(NUMERIC_COLUMNS)}
    missing.update({c: int(np.sum(categorical[:, j] == "missing")) for j, c in enumerate(CATEGORICAL_COLUMNS)})
    return {"numeric": numeric, "categorical": categorical, "y": y, "classes": classes.tolist(),
            "info": {"rows": len(y), "class_balance": {str(c): int(np.sum(y == j)) for j, c in enumerate(classes)}, "missing_values": missing}}


def fit_transform(data: dict, train: np.ndarray, query: np.ndarray) -> tuple:
    numeric, categorical = data["numeric"], data["categorical"]
    median = np.nanmedian(numeric[train], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    A = np.where(np.isnan(numeric[train]), median, numeric[train])
    B = np.where(np.isnan(numeric[query]), median, numeric[query])
    mean, scale = A.mean(axis=0), A.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    left, right = [(A - mean) / scale], [(B - mean) / scale]
    names = NUMERIC_COLUMNS.copy()
    for j, column in enumerate(CATEGORICAL_COLUMNS):
        categories = np.unique(categorical[train, j])
        left.append((categorical[train, j, None] == categories).astype(float))
        right.append((categorical[query, j, None] == categories).astype(float))
        names.extend(f"{column}={c}" for c in categories)
    return np.column_stack(left), np.column_stack(right), names


def loo_distances(data: dict, subset: np.ndarray) -> np.ndarray:
    D = np.full((len(subset), len(subset)), np.inf)
    for i, obj in enumerate(subset):
        mask = np.arange(len(subset)) != i
        A, B, _ = fit_transform(data, subset[mask], np.array([obj]))
        D[i, mask] = np.linalg.norm(A - B, axis=1)
    return D


def distances(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    squares = np.sum(X * X, axis=1)[:, None] + np.sum(Z * Z, axis=1)[None, :] - 2.0 * X @ Z.T
    return np.sqrt(np.maximum(squares, 0.0))


def neighbor_order(D: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    D = D.copy()
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    indices = np.argsort(D, axis=1, kind="stable")
    return indices, np.take_along_axis(D, indices, axis=1)


def gaussian(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u * u)


def predict_parzen_from_neighbors(
    indices: np.ndarray, sorted_distances: np.ndarray, y_reference: np.ndarray, k: int, n_classes: int
) -> np.ndarray:
    k = min(k, indices.shape[1] - 1)
    h = np.maximum(sorted_distances[:, k], 1e-12)
    chosen = indices[:, :k]
    weights = gaussian(sorted_distances[:, :k] / h[:, None])
    scores = np.zeros((len(indices), n_classes), dtype=float)
    rows = np.repeat(np.arange(len(indices)), k)
    np.add.at(scores, (rows, y_reference[chosen].reshape(-1)), weights.reshape(-1))
    return scores.argmax(axis=1)


def predict_parzen(X: np.ndarray, X_reference: np.ndarray, y_reference: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    indices, ordered = neighbor_order(distances(X, X_reference))
    return predict_parzen_from_neighbors(indices, ordered, y_reference, k, n_classes)


def predict_knn_from_neighbors(indices: np.ndarray, y_reference: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    neighbors = y_reference[indices[:, :k]]
    counts = np.stack([np.sum(neighbors == c, axis=1) for c in range(n_classes)], axis=1)
    return counts.argmax(axis=1)


def loo_risks(D: np.ndarray, y: np.ndarray, k_values: list[int], n_classes: int) -> dict:
    indices, ordered = neighbor_order(D)
    ordinary = []
    parzen = []
    for k in k_values:
        ordinary.append(float(np.mean(predict_knn_from_neighbors(indices, y, k, n_classes) != y)))
        parzen.append(float(np.mean(predict_parzen_from_neighbors(indices, ordered, y, k, n_classes) != y)))
    return {"k": k_values, "knn": ordinary, "parzen": parzen}


def predict_fixed_parzen(D: np.ndarray, labels: np.ndarray, bandwidth: float, n_classes: int) -> np.ndarray:
    weights = gaussian(D / bandwidth)
    votes = np.column_stack([weights[:, labels == c].sum(axis=1) for c in range(n_classes)])
    return votes.argmax(axis=1)


def fixed_bandwidth_risks(D: np.ndarray, y: np.ndarray, bandwidths: list[float], n_classes: int) -> list[float]:
    return [float(np.mean(predict_fixed_parzen(D, y, h, n_classes) != y)) for h in bandwidths]


def choose_k(values: list[float], k_values: list[int]) -> int:
    return int(k_values[int(np.argmin(values))])


def compactness_profile(X: np.ndarray, y: np.ndarray, max_m: int = 30) -> list[float]:
    indices, _ = neighbor_order(distances(X, X), exclude_self=True)
    return np.mean(y[indices[:, :max_m]] == y[:, None], axis=0).tolist()


def backward_path(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    D = distances(X, X)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    rows = np.arange(len(y))
    first, second = order[:, 0].copy(), order[:, 1].copy()
    second_pos = np.ones(len(y), dtype=int)
    mask = np.ones(len(y), dtype=bool)
    removed = []
    history = [{"n": len(y), "selection_error": float(np.mean(y[first] != y))}]
    while np.sum(mask) > len(np.unique(y)) + 1:
        baseline = int(np.sum(y[first] != y))
        delta = (y[second] != y).astype(int) - (y[first] != y).astype(int)
        costs = np.bincount(first, weights=delta, minlength=len(y))
        counts = np.bincount(y[mask], minlength=int(y.max()) + 1)
        costs[~mask | (counts[y] <= 1)] = np.inf
        chosen = int(np.argmin(costs))
        if not np.isfinite(costs[chosen]) or costs[chosen] > 0:
            break
        mask[chosen] = False
        removed.append(chosen)
        history.append({"n": int(np.sum(mask)), "selection_error": float((baseline + costs[chosen]) / len(y))})
        affected = rows[(first == chosen) | (second == chosen)]
        shift = rows[first == chosen]
        first[shift] = second[shift]
        second_pos[affected] += 1
        pending = affected[~mask[order[affected, second_pos[affected]]]]
        while len(pending):
            second_pos[pending] += 1
            pending = pending[~mask[order[pending, second_pos[pending]]]]
        second[affected] = order[affected, second_pos[affected]]
    return np.array(removed, dtype=int), history


def path_subset(n: int, removed: np.ndarray, fraction: float) -> np.ndarray:
    target = max(1, (n * round(100 * fraction) + 99) // 100)
    steps = min(len(removed), n - target)
    mask = np.ones(n, dtype=bool)
    mask[removed[:steps]] = False
    return np.flatnonzero(mask)


def backward_selection(X: np.ndarray, y: np.ndarray, fraction: float = 0.0) -> tuple[np.ndarray, list[dict]]:
    removed, history = backward_path(X, y)
    return path_subset(len(y), removed, fraction), history


def forward_selection(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    D = distances(X, X)
    np.fill_diagonal(D, np.inf)
    selected = []
    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        center = np.mean(X[class_indices], axis=0)
        selected.append(int(class_indices[np.argmin(np.sum((X[class_indices] - center) ** 2, axis=1))]))
    selected_mask = np.zeros(len(y), dtype=bool)
    selected_mask[selected] = True
    initial = np.flatnonzero(selected_mask)
    history = [{"n": len(initial), "selection_error": float(np.mean(y[initial[np.argmin(D[:, initial], axis=1)]] != y))}]
    while np.sum(selected_mask) < len(y) - 1:
        work = D.copy()
        work[:, ~selected_mask] = np.inf
        nearest = np.argmin(work, axis=1)
        nearest_dist = work[np.arange(len(y)), nearest]
        baseline_correct = y[nearest] == y
        baseline_error = int(np.sum(~baseline_correct))
        candidates = np.flatnonzero(~selected_mask)
        gains = np.empty(len(candidates), dtype=int)
        for j, candidate in enumerate(candidates):
            improved = (D[:, candidate] < nearest_dist) | ((D[:, candidate] == nearest_dist) & (candidate < nearest))
            gains[j] = int(np.sum(improved & (y[candidate] == y) & ~baseline_correct) - np.sum(improved & (y[candidate] != y) & baseline_correct))
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            break
        selected_mask[candidates[j]] = True
        history.append({"n": int(np.sum(selected_mask)), "selection_error": float((baseline_error - gains[j]) / len(y))})
    return np.flatnonzero(selected_mask), history


def scores(y: np.ndarray, prediction: np.ndarray, n_classes: int) -> dict:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (y, prediction), 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(np.mean(prediction == y)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }


def pca_coordinates(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return centered @ right[:2].T


def plot_risks(risks: dict, k_knn: int, k_parzen: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(risks["k"], risks["knn"], marker=".", label="Метод k ближайших соседей")
    ax.plot(risks["k"], risks["parzen"], marker=".", label="Окно Парзена, гауссово ядро")
    ax.axvline(k_knn, linestyle="--", alpha=0.5)
    ax.axvline(k_parzen, linestyle=":", alpha=0.5)
    ax.set(
        xlabel="Количество соседей k",
        ylabel="Эмпирический риск LOO",
        title="Зависимость эмпирического риска от количества соседей",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "loo_risk.png", dpi=160)
    plt.close(fig)


def plot_bandwidths(bandwidths: list[float], risks: list[float], chosen: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bandwidths, risks, marker="o")
    ax.axvline(chosen, linestyle="--", alpha=0.5)
    ax.set(
        xlabel="Ширина окна h",
        ylabel="Эмпирический риск LOO",
        title="Метод окна Парзена с фиксированной шириной",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "bandwidth_risk.png", dpi=160)
    plt.close(fig)


def plot_compactness(profile: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(profile) + 1), profile, marker=".")
    ax.set(
        xlabel="Порядковый номер соседа m",
        ylabel="Профиль компактности Π(m)",
        title="Профиль компактности классов",
        ylim=(max(0, min(profile) - 0.01), 1.0),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "compactness.png", dpi=160)
    plt.close(fig)


def plot_prototypes(X: np.ndarray, y: np.ndarray, classes: list[str], selected: dict[str, np.ndarray]) -> None:
    coords = pca_coordinates(X)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    names = {
        "All objects": "Все объекты",
        "backward": "Последовательное удаление",
        "forward": "Последовательное добавление",
    }
    for ax, (name, chosen) in zip(axes, [("All objects", np.arange(len(y))), *selected.items()]):
        for label, species in enumerate(classes):
            mask = y == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=11, alpha=0.16, label=species, color=f"C{label}")
            relevant = chosen[y[chosen] == label]
            ax.scatter(coords[relevant, 0], coords[relevant, 1], s=35, edgecolors="black", linewidths=0.3, color=f"C{label}")
        ax.set(
            title=f"{names.get(name, name)} (объектов: {len(chosen)})",
            xlabel="Первая главная компонента",
            ylabel="Вторая главная компонента",
        )
        ax.grid(alpha=0.15)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "prototypes_pca.png", dpi=160)
    plt.close(fig)


def plot_selection_histories(histories: dict[str, list[dict]], baseline_error: float, n: int, selected_n: int) -> None:
    names = {
        "backward": ("Последовательное удаление эталонов", "prototype_backward.png"),
        "forward": ("Последовательное добавление эталонов", "prototype_forward.png"),
    }
    for name, (title, filename) in names.items():
        entries = histories[name]
        fig, ax = plt.subplots(figsize=(8, 5))
        if entries:
            ax.plot(
                [v["n"] for v in entries],
                [v["selection_error"] for v in entries],
                marker=".",
                label="Ошибка после отбора",
            )
        ax.set(
            xlabel="Количество эталонов",
            ylabel="Внутренний критерий отбора (1-NN)",
            title=title,
        )
        if name == "backward":
            ax.axvline(selected_n, color="C1", linestyle="--", label=f"Выбрано: {selected_n}")
            ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_PATH / filename, dpi=160)
        plt.close(fig)


def reference_loo(data: dict, subset: np.ndarray, k: int) -> np.ndarray:
    y = data["y"]
    prediction = np.empty(len(subset), dtype=int)
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform")
    for i, obj in enumerate(subset):
        train = np.delete(subset, i)
        A, B, _ = fit_transform(data, train, np.array([obj]))
        model.fit(A, y[train])
        prediction[i] = model.predict(B)[0]
    return prediction


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        for j, part in enumerate(np.array_split(rng.permutation(np.flatnonzero(y == label)), n_folds)):
            groups[j].extend(part.tolist())
    return [np.array(sorted(group), dtype=int) for group in groups]


def predict_knn(X: np.ndarray, Z: np.ndarray, y: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    indices, _ = neighbor_order(distances(X, Z))
    return predict_knn_from_neighbors(indices, y, k, n_classes)


def benchmark_prediction(X, Z, y, k, n_classes) -> float:
    predict_knn(X, Z, y, k, n_classes)
    times = []
    for _ in range(7):
        start = time.perf_counter()
        predict_knn(X, Z, y, k, n_classes)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def candidate_predictions(B, A, labels, chosen, n_classes):
    indices, _ = neighbor_order(distances(B, A[chosen]))
    limit = min(max(K_VALUES), len(chosen))
    neighbors = labels[chosen][indices[:, :limit]]
    votes = np.cumsum(neighbors[:, :, None] == np.arange(n_classes), axis=1)
    return votes.argmax(axis=2)


def tune_selection(data: dict, subset: np.ndarray, seed: int = SEED) -> dict:
    y = data["y"]
    n_classes = len(data["classes"])
    candidates = {str(f): [] for f in RETAINED_FRACTIONS}
    candidates["forward"] = []
    for val_local in stratified_folds(y[subset], INNER_FOLDS, seed):
        train_local = np.setdiff1d(np.arange(len(subset)), val_local)
        train, val = subset[train_local], subset[val_local]
        A, B, _ = fit_transform(data, train, val)
        labels = y[train]
        removed, _ = backward_path(A, labels)
        choices = {str(f): path_subset(len(train), removed, f) for f in RETAINED_FRACTIONS}
        choices["forward"], _ = forward_selection(A, labels)
        for key, chosen in choices.items():
            pred = candidate_predictions(B, A, labels, chosen, n_classes)
            candidates[key].append((y[val], pred, len(chosen)))
    results = {}
    for key, folds in candidates.items():
        common_k = min(pred.shape[1] for _, pred, _ in folds)
        truth = np.concatenate([truth for truth, _, _ in folds])
        pred = np.concatenate([pred[:, :common_k] for _, pred, _ in folds])
        risk = np.mean(pred != truth[:, None], axis=0)
        k = int(np.argmin(risk)) + 1
        results[key] = {"k": k, "risk": float(risk[k-1]),
                        "macro_f1": scores(truth, pred[:, k-1], n_classes)["macro_f1"],
                        "mean_n_prototypes": float(np.mean([n for _, _, n in folds])),
                        "n_prototypes_by_fold": [n for _, _, n in folds],
                        "fold_risks": [float(np.mean(p[:, k-1] != t)) for t, p, _ in folds]}
    threshold = results["1.0"]["risk"] + MAX_RISK_INCREASE
    eligible = [f for f in RETAINED_FRACTIONS if results[str(f)]["risk"] <= threshold + 1e-12]
    fraction = min(eligible, key=lambda f: (results[str(f)]["mean_n_prototypes"], results[str(f)]["risk"], -f))
    return {"inner_folds": INNER_FOLDS, "seed": seed, "epsilon": MAX_RISK_INCREASE,
            "threshold": threshold, "selected_fraction": fraction, "candidates": results}


def external_cv(data: dict, n_folds: int = 5, seed: int = 42) -> dict:
    y = data["y"]
    n_classes = len(data["classes"])
    methods = {"all": None, "backward": backward_selection, "forward": forward_selection}
    predictions = {name: {mode: np.empty(len(y), dtype=int) for mode in ("k1", "tuned")} for name in methods}
    records = []
    for fold, test in enumerate(stratified_folds(y, n_folds, seed), 1):
        train = np.setdiff1d(np.arange(len(y)), test)
        A, B, _ = fit_transform(data, train, test)
        labels = y[train]
        tuning_start = time.perf_counter()
        tuning = tune_selection(data, train, seed)
        record = {"fold": fold, "n_train": len(train), "n_test": len(test), "methods": {},
                  "selection_tuning": tuning, "tuning_seconds": time.perf_counter() - tuning_start}
        for name, algorithm in methods.items():
            start = time.perf_counter()
            if algorithm is None:
                chosen = np.arange(len(train))
                selection_time = 0.0
                k = tuning["candidates"]["1.0"]["k"]
            elif name == "backward":
                fraction = tuning["selected_fraction"]
                chosen, _ = backward_selection(A, labels, fraction)
                selection_time = time.perf_counter() - start
                k = min(tuning["candidates"][str(fraction)]["k"], len(chosen))
            else:
                chosen, _ = algorithm(A, labels)
                selection_time = time.perf_counter() - start
                k = min(tuning["candidates"]["forward"]["k"], len(chosen))
            item = {"n_prototypes": len(chosen), "retained_fraction": len(chosen) / len(train), "k": k,
                    "selection_seconds": selection_time,
                    "prediction_seconds_k1": benchmark_prediction(B, A[chosen], labels[chosen], 1, n_classes),
                    "prediction_seconds_tuned": benchmark_prediction(B, A[chosen], labels[chosen], k, n_classes)}
            for mode, local_k in [("k1", 1), ("tuned", k)]:
                pred = predict_knn(B, A[chosen], labels[chosen], local_k, n_classes)
                predictions[name][mode][test] = pred
                item[mode] = scores(y[test], pred, n_classes)
            record["methods"][name] = item
        records.append(record)
        print(f"External fold {fold}/{n_folds} completed", flush=True)
    summary = {}
    for name in methods:
        items = [r["methods"][name] for r in records]
        summary[name] = {"mean_n_prototypes": float(np.mean([r["n_prototypes"] for r in items])),
                         "mean_retained_fraction": float(np.mean([r["retained_fraction"] for r in items])),
                         "selected_k": [r["k"] for r in items],
                         "total_selection_seconds": float(sum(r["selection_seconds"] for r in items)),
                         "prediction_seconds_per_object_k1": float(sum(r["prediction_seconds_k1"] for r in items) / len(y)),
                         "prediction_seconds_per_object_tuned": float(sum(r["prediction_seconds_tuned"] for r in items) / len(y))}
        for mode in ("k1", "tuned"):
            summary[name][mode] = scores(y, predictions[name][mode], n_classes)
            summary[name][mode]["fold_accuracy_std"] = float(np.std([r[mode]["accuracy"] for r in items]))
    return {"n_folds": n_folds, "seed": seed, "folds": records, "summary": summary}


def plot_external(cv: dict) -> None:
    names = ["all", "backward", "forward"]
    labels = ["Все объекты", "Удаление", "Добавление"]
    records = cv["summary"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    for offset, mode, label in [(-0.18, "k1", "k = 1"), (0.18, "tuned", "k подобрано")]:
        errors = [100 * (1 - records[n][mode]["accuracy"]) for n in names]
        bars = ax.bar(x + offset, errors, width=0.35, label=label)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Ошибка на внешних фолдах, %", title="Обычный kNN до и после отбора эталонов")
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "prototype_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    times = [records[n]["prediction_seconds_per_object_k1"] * 1e6 for n in names]
    bars = ax.bar(labels, times)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(ylabel="Микросекунд на объект", title="Время предсказания 1-NN (пакетный режим)")
    ax.set_yscale("log")
    ax.set_ylim(min(times) / 2, max(times) * 2)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "prediction_time.png", dpi=160)
    plt.close(fig)


def plot_tuning(tuning: dict) -> None:
    records = tuning["candidates"]
    labels = [f"{f * 100:g}%" if f else "До остановки" for f in RETAINED_FRACTIONS]
    values = [records[str(f)] for f in RETAINED_FRACTIONS]
    pos = np.arange(len(values))
    selected = RETAINED_FRACTIONS.index(tuning["selected_fraction"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    risk = [100 * v["risk"] for v in values]
    axes[0].plot(pos, risk, marker="o", label="Ошибка внутреннего CV")
    axes[0].axhline(100 * tuning["threshold"], linestyle="--", color="C1", label="Допустимая ошибка")
    axes[0].scatter(selected, risk[selected], s=95, facecolors="none", edgecolors="C3", linewidths=2, label="Выбранный вариант")
    axes[0].set_ylabel("Ошибка, %")
    axes[0].legend()
    axes[1].plot(pos, [v["mean_n_prototypes"] for v in values], marker="o")
    axes[1].set_ylabel("Среднее число эталонов")
    axes[1].set_xticks(pos, labels, rotation=40)
    axes[1].set_xlabel("Целевая доля оставленных эталонов")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Подбор объёма эталонов: внутренний стратифицированный 3-fold CV")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "prototype_tuning.png", dpi=160)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    FIGURE_PATH.mkdir(exist_ok=True)
    data = read_dataset(find_dataset())
    y = data["y"]
    all_indices = np.arange(len(y))
    X, _, names = fit_transform(data, all_indices, all_indices)
    data["info"]["features_after_encoding"] = X.shape[1]
    n_classes = len(data["classes"])
    D = loo_distances(data, all_indices)
    risks = loo_risks(D, y, K_VALUES, n_classes)
    k_knn = choose_k(risks["knn"], K_VALUES)
    k_parzen = choose_k(risks["parzen"], K_VALUES)
    fixed_risks = fixed_bandwidth_risks(D, y, BANDWIDTHS, n_classes)
    fixed_h = BANDWIDTHS[int(np.argmin(fixed_risks))]
    profile = compactness_profile(X, y)
    indices, ordered = neighbor_order(D)
    own_knn = predict_knn_from_neighbors(indices, y, k_knn, n_classes)
    full_parzen = predict_parzen_from_neighbors(indices, ordered, y, k_parzen, n_classes)
    fixed_parzen = predict_fixed_parzen(D, y, fixed_h, n_classes)
    reference_pred = reference_loo(data, all_indices, k_knn)
    print("Full-sample LOO completed", flush=True)
    tuning = tune_selection(data, all_indices)
    print(f"Full-sample tuning: retained fraction = {tuning['selected_fraction']}", flush=True)
    selections = {}
    histories = {}
    for name, algorithm in [("backward", backward_selection), ("forward", forward_selection)]:
        chosen, history = algorithm(X, y, tuning["selected_fraction"]) if name == "backward" else algorithm(X, y)
        histories[name] = history
        selections[name] = {"n_prototypes": len(chosen), "retained_fraction": len(chosen) / len(y), "indices": chosen.tolist()}
    selections["backward"].update({"selected_fraction": tuning["selected_fraction"],
        "selected_k": tuning["candidates"][str(tuning["selected_fraction"])]["k"],
        "removed_count": len(y) - selections["backward"]["n_prototypes"],
        "natural_stop_n": histories["backward"][-1]["n"]})
    cv = external_cv(data)
    plot_risks(risks, k_knn, k_parzen)
    plot_bandwidths(BANDWIDTHS, fixed_risks, fixed_h)
    plot_compactness(profile)
    plot_prototypes(X, y, data["classes"], {name: np.array(record["indices"]) for name, record in selections.items()})
    plot_selection_histories(histories, risks["knn"][0], len(y), selections["backward"]["n_prototypes"])
    plot_tuning(tuning)
    plot_external(cv)
    results = {"dataset": "srisyra02/penguin-species-dataset", "data": data["info"], "classes": data["classes"], "feature_names": names,
        "distance": "Euclidean; preprocessing fitted on training objects in each evaluation split",
        "kernel": "Gaussian exp(-u²/2)", "bandwidth": "distance to the (k+1)-th neighbor; truncated Gaussian over first k neighbors",
        "evaluation": {"loo": "Full-sample LOO for k and h tuning; scores at the minimum are descriptive, not independent",
                       "external_cv": "Stratified 5-fold CV with nested stratified 3-fold tuning of retained fraction and k; preprocessing and selection refitted within each split",
                       "full_sample_selection": "Descriptive visualization only; not used for external predictions"},
        "loo": risks, "fixed_bandwidth_loo": {"h": BANDWIDTHS, "risk": fixed_risks, "selected_h": fixed_h},
        "selected_k": {"knn": k_knn, "parzen": k_parzen}, "compactness_profile": profile,
        "baseline": {"own_knn": scores(y, own_knn, n_classes), "sklearn_knn": scores(y, reference_pred, n_classes),
                     "parzen": scores(y, full_parzen, n_classes), "fixed_parzen": scores(y, fixed_parzen, n_classes),
                     "sklearn_agreement": float(np.mean(own_knn == reference_pred))},
        "prototype_selection": selections, "selection_histories": histories, "selection_tuning": tuning, "external_cv": cv,
        "runtime_seconds": float(time.perf_counter() - started)}
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
