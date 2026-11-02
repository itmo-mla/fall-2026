import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import svd
from sklearn.decomposition import PCA

from data import PLOTS_DIR, as_numpy, load_raw_dataframe, prepare_data

VARIANCE_LEVELS = (0.90, 0.95)
KAISER_THRESHOLD = 1.0

FEATURE_LABELS = {
    "age": "возраст",
    "sleep_hours": "сон",
    "physical_activity": "активность",
    "caffeine_mg": "кофеин",
    "alcohol_drinks": "алкоголь",
    "diet_quality": "диета",
    "stress_level": "стресс",
    "heart_rate": "пульс",
    "breathing_rate": "дыхание",
    "sweating_level": "потливость",
    "therapy_sessions": "терапия",
    "smoking": "курение",
    "family_history": "семья",
    "dizziness": "головокружение",
    "medication": "лекарства",
    "major_life_event": "событие",
    "gender_Male": "пол: мужской",
    "gender_Other": "пол: другой",
}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("occupation_"):
        return "проф. " + name.removeprefix("occupation_")
    return name


class SVDPrincipalComponents:
    def fit(self, X: np.ndarray) -> "SVDPrincipalComponents":
        matrix = np.asarray(X, dtype=float)
        n_samples, _ = matrix.shape
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        U, singular_values, components = svd(centered, full_matrices=False)
        components = self._fix_signs(U, components)

        explained_variance = (singular_values ** 2) / (n_samples - 1)
        total = explained_variance.sum()
        self.components_ = components
        self.singular_values_ = singular_values
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance / total
        self.n_samples_ = n_samples
        self.n_features_ = matrix.shape[1]
        return self

    @staticmethod
    def _fix_signs(scores_u: np.ndarray, components: np.ndarray) -> np.ndarray:
        largest = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(components.shape[0]), largest])
        signs[signs == 0.0] = 1.0
        scores_u *= signs
        return components * signs[:, np.newaxis]

    def transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        centered = np.asarray(X, dtype=float) - self.mean_
        axes = self.components_ if n_components is None else self.components_[:n_components]
        return centered @ axes.T


def components_for_variance(ratio: np.ndarray, level: float) -> int:
    cumulative = np.cumsum(ratio)
    return int(np.searchsorted(cumulative, level, side="left") + 1)


def participation_ratio(explained_variance: np.ndarray) -> float:
    total = explained_variance.sum()
    return float(total ** 2 / np.sum(explained_variance ** 2))


def align_to_reference(components: np.ndarray, reference: np.ndarray) -> np.ndarray:
    signs = np.sign(np.sum(components * reference, axis=1))
    signs[signs == 0.0] = 1.0
    return components * signs[:, np.newaxis]


def compare_with_sklearn(model: SVDPrincipalComponents, X_train: np.ndarray, X_test: np.ndarray) -> dict:
    full = PCA(svd_solver="full").fit(X_train)
    auto = PCA(svd_solver="auto").fit(X_train)
    own_components = align_to_reference(model.components_, full.components_)
    own_scores_train = X_train - model.mean_
    own_scores_train = own_scores_train @ own_components.T
    own_scores_test = (X_test - model.mean_) @ own_components.T

    def gap(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.max(np.abs(left - right)))

    full_train = full.transform(X_train)
    full_test = full.transform(X_test)
    auto_components = align_to_reference(auto.components_, full.components_)
    return {
        "auto_solver": auto._fit_svd_solver,
        "singular_values": gap(model.singular_values_, full.singular_values_),
        "explained_variance_ratio": gap(model.explained_variance_ratio_, full.explained_variance_ratio_),
        "components": gap(own_components, full.components_),
        "train_scores": gap(own_scores_train, full_train),
        "test_scores": gap(own_scores_test, full_test),
        "auto_ratio": gap(model.explained_variance_ratio_, auto.explained_variance_ratio_),
        "auto_components": gap(own_components, auto_components),
        "auto_test_scores": gap(
            own_scores_test,
            (X_test - auto.mean_) @ auto_components.T,
        ),
    }


def score_target_correlation(scores: np.ndarray, y: np.ndarray, limit: int = 8) -> np.ndarray:
    correlations = []
    for column in range(limit):
        correlations.append(float(np.corrcoef(scores[:, column], y)[0, 1]))
    return np.asarray(correlations)


def _plot_scree(model: SVDPrincipalComponents, kaiser_count: int, k90: int) -> None:
    index = np.arange(1, model.n_features_ + 1)
    ratio = model.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(index, model.explained_variance_, color="#4C72B0", label="дисперсия компоненты")
    axes[0].axhline(KAISER_THRESHOLD, color="#C44E52", linewidth=1.2, label="порог Кайзера, λ = 1")
    axes[0].set_xlim(0.4, 30.6)
    axes[0].set_title("Дисперсия главных компонент")
    axes[0].set_xlabel("Номер компоненты")
    axes[0].set_ylabel("Дисперсия, λ = s² / (n − 1)")
    axes[0].legend()

    axes[1].plot(index, cumulative, color="#4C72B0", marker="o", markersize=3, label="накопленная доля")
    axes[1].axhline(0.90, color="#C44E52", linewidth=1.2, label="90% дисперсии")
    axes[1].axvline(k90, color="#55A868", linestyle="--", linewidth=1.2, label=f"{k90} компонент до 90%")
    axes[1].set_title("Сколько компонент держат дисперсию")
    axes[1].set_xlabel("Число компонент")
    axes[1].set_ylabel("Накопленная доля дисперсии")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    for ax in axes:
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

    fig.suptitle(
        f"Спектр PCA: выше λ = 1 только {kaiser_count} компонент из {model.n_features_}"
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_scree.png", dpi=120)
    plt.close(fig)


def _plot_loadings(
    model: SVDPrincipalComponents,
    feature_names: list[str],
    count: int,
) -> None:
    loadings = model.components_[:count]
    labels = [feature_label(name) for name in feature_names]
    limit = float(np.max(np.abs(loadings)))
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(loadings.T, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(count))
    ax.set_xticklabels([f"PC{i}" for i in range(1, count + 1)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Нагрузки первых {count} компонент")
    ax.set_xlabel("Компонента")
    ax.set_ylabel("Признак")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("вес признака в компоненте")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_loadings.png", dpi=120)
    plt.close(fig)


def _plot_scores(scores: np.ndarray, y: np.ndarray, ratio: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cloud = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=y,
        cmap="viridis",
        s=8,
        alpha=0.45,
        label="человек из train",
    )
    colorbar = fig.colorbar(cloud, ax=ax)
    colorbar.set_label("балл тревожности")
    ax.set_title("Train в плоскости первых двух компонент")
    ax.set_xlabel(f"PC1, {ratio[0] * 100:.1f}% дисперсии")
    ax.set_ylabel(f"PC2, {ratio[1] * 100:.1f}% дисперсии")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C72B0", markersize=6, label="человек из train")])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "09_pc_plane.png", dpi=120)
    plt.close(fig)


def print_report(
    model: SVDPrincipalComponents,
    feature_names: list[str],
    correlations: np.ndarray,
    gaps: dict,
) -> None:
    ratio = model.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    kaiser = int(np.sum(model.explained_variance_ > KAISER_THRESHOLD))
    participation = participation_ratio(model.explained_variance_)
    gaps_between = model.singular_values_[:-1] / model.singular_values_[1:]

    print("\n=== Спектр ===")
    print(f"компонент: {model.n_features_}, объектов train: {model.n_samples_}")
    print("доля первой компоненты:", round(float(ratio[0]), 4))
    print("равная доля 1/p:", round(1 / model.n_features_, 4))
    print("λ min/max:", round(float(model.explained_variance_.min()), 4), round(float(model.explained_variance_.max()), 4))
    print("Кайзер λ>1:", kaiser)
    for level in VARIANCE_LEVELS:
        k = components_for_variance(ratio, level)
        print(f"компонент до {level:.0%}: {k}, фактическая доля {cumulative[k - 1]:.4f}")
    print("коэффициент участия:", round(participation, 2))
    gap_at = int(np.argmax(gaps_between))
    print(
        f"максимальный разрыв s_{gap_at + 1}/s_{gap_at + 2}:",
        round(float(gaps_between[gap_at]), 4),
    )
    print("накопленная доля на Кайзере:", round(float(cumulative[kaiser - 1]), 4))
    print("минимальное сингулярное число:", float(model.singular_values_.min()))

    print("\n=== Нагрузки первых компонент ===")
    show = max(kaiser, 2)
    for component in range(show):
        order = np.argsort(np.abs(model.components_[component]))[::-1]
        top = ", ".join(
            f"{feature_label(feature_names[position])} {model.components_[component, position]:+.2f}"
            for position in order[:3]
        )
        print(
            f"PC{component + 1}: λ={model.explained_variance_[component]:.3f}, "
            f"доля={ratio[component]:.3f}, {top}"
        )

    print("\n=== Корреляция проекции с баллом тревожности ===")
    for index, value in enumerate(correlations, start=1):
        print(f"PC{index}: {value:.3f}")

    print("\n=== Расхождение со sklearn, max |Δ| ===")
    print("solver auto:", gaps["auto_solver"])
    for key, value in gaps.items():
        if key == "auto_solver":
            continue
        print(f"{key}: {value:.3e}")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    prepared = prepare_data(load_raw_dataframe())
    X_train, X_test, y_train, _ = as_numpy(prepared)
    model = SVDPrincipalComponents().fit(X_train)
    scores = model.transform(X_train)
    correlations = score_target_correlation(scores, y_train)
    gaps = compare_with_sklearn(model, X_train, X_test)
    if gaps["components"] > 1e-8 or gaps["test_scores"] > 1e-8:
        raise RuntimeError(f"Своё PCA разошлось со sklearn SVD сильнее 1e-8: {gaps}")

    kaiser = int(np.sum(model.explained_variance_ > KAISER_THRESHOLD))
    k90 = components_for_variance(model.explained_variance_ratio_, 0.90)
    loading_count = min(8, max(kaiser, 2))
    _plot_scree(model, kaiser, k90)
    _plot_loadings(model, prepared["feature_names"], loading_count)
    _plot_scores(scores, y_train, model.explained_variance_ratio_)
    print_report(model, prepared["feature_names"], correlations, gaps)
    print(f"\nГрафики: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
