import math

import numpy as np


def pairwise_euclidean(
        x: np.ndarray,
        y: np.ndarray
) -> np.ndarray:

    x_norm = np.sum(x ** 2, axis=1, keepdims=True)
    y_norm = np.sum(y ** 2, axis=1, keepdims=True).T

    squared = x_norm + y_norm - 2.0 * x @ y.T
    squared = np.maximum(squared, 0.0)

    return np.sqrt(squared)


def gaussian_kernel(r: np.ndarray) -> np.ndarray:

    weights = np.exp(-0.5 * r ** 2)
    return np.where(r <= 1.0, weights, 0.0)


class KNNClassifier:
    def __init__(
            self,
            k: int = 5,
            eps: float = 1e-12
    ):

        self.k = k
        self.eps = eps

        self.x_train = None
        self.y_train = None
        self.classes_ = None

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> "KNNClassifier":

        self.x_train = x
        self.y_train = y
        self.classes_ = np.unique(y)

        return self

    def _bandwidth(self, distances: np.ndarray) -> np.ndarray:
        n_train = distances.shape[1]
        neighbor_index = min(self.k, n_train - 1)

        h = np.partition(
            distances,
            kth=neighbor_index,
            axis=1
        )[:, neighbor_index]

        zero_mask = h <= self.eps

        if np.any(zero_mask):
            positive_distances = np.where(
                distances > self.eps,
                distances,
                np.inf
            )

            fallback = np.min(positive_distances, axis=1)

            fallback = np.where(
                np.isfinite(fallback),
                fallback,
                1.0
            )

            h = np.where(zero_mask, fallback, h)

        return np.maximum(h, self.eps)

    def _class_scores_from_distances(
            self,
            distances: np.ndarray
    ) -> np.ndarray:
        h = self._bandwidth(distances)

        normalized = distances / h[:, None]
        weights = gaussian_kernel(normalized)

        scores = np.zeros(
            (
                len(distances),
                len(self.classes_),
            ),
            dtype=float,
        )

        for class_index, class_label in enumerate(self.classes_):
            class_mask = self.y_train == class_label
            scores[:, class_index] = np.sum(
                weights[:, class_mask],
                axis=1
            )

        return scores

    def predict_proba(
            self,
            x: np.ndarray
    ) -> np.ndarray:

        distances = pairwise_euclidean(x, self.x_train)
        scores = self._class_scores_from_distances(distances)
        score_sum = np.sum(scores, axis=1, keepdims=True)

        probabilities = np.divide(
            scores,
            score_sum,
            out=np.full_like(scores, 1.0 / len(self.classes_)),
            where=score_sum > self.eps,
        )

        return probabilities

    def predict(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        probabilities = self.predict_proba(x)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]


def loo_risk_curve(
        x: np.ndarray,
        y: np.ndarray,
        k_values,
        eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    k_values = np.asarray(list(k_values), dtype=int)

    valid_k = k_values[(k_values >= 1) & (k_values <= len(x) - 2)]

    distances = pairwise_euclidean(x, x)
    np.fill_diagonal(distances, np.inf)

    classes = np.unique(y)
    risks = []

    for k in valid_k:
        h = np.partition(
            distances,
            kth=k,
            axis=1,
        )[:, k]

        zero_mask = h <= eps

        if np.any(zero_mask):
            positive_distances = np.where(
                distances > eps,
                distances,
                np.inf
            )

            fallback = np.min(positive_distances, axis=1)

            fallback = np.where(
                np.isfinite(fallback),
                fallback,
                1.0
            )

            h = np.where(
                zero_mask,
                fallback,
                h
            )

        normalized = distances / np.maximum(h[:, None], eps)
        weights = gaussian_kernel(normalized)

        class_scores = np.zeros(
            (
                len(x),
                len(classes),
            ),
            dtype=float,
        )

        for class_index, class_label in enumerate(classes):
            class_scores[:, class_index] = np.sum(
                weights[:, y == class_label],
                axis=1
            )

        prediction = classes[np.argmax(class_scores, axis=1)]
        risk = np.mean(prediction != y)
        risks.append(float(risk))

    return valid_k, np.asarray(risks, dtype=float)


def select_best_k_loo(
        x: np.ndarray,
        y: np.ndarray,
        k_values,
) -> tuple[int, np.ndarray, np.ndarray]:
    k_values, risks = loo_risk_curve(
        x=x,
        y=y,
        k_values=k_values,
    )

    best_index = int(np.argmin(risks))

    return (
        int(k_values[best_index]),
        k_values,
        risks,
    )


def compactness_profile(
        x: np.ndarray,
        y: np.ndarray,
        max_m: int | None = None,
) -> np.ndarray:

    distances = pairwise_euclidean(x, x)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1)

    max_possible = len(x) - 1

    if max_m is None:
        max_m = max_possible
    else:
        max_m = min(int(max_m), max_possible)

    neighbor_labels = y[order[:, :max_m]]
    profile = np.mean(
        neighbor_labels != y[:, None],
        axis=0
    )

    return profile.astype(float)


def _ccv_weights(
        total_size: int,
        control_size: int
) -> np.ndarray:
    L = int(total_size)
    k_control = int(control_size)
    train_size = L - k_control

    denominator = math.comb(L - 1, train_size)
    weights = []

    for m in range(1, k_control + 1):
        upper = L - 1 - m

        if upper < train_size - 1:
            weight = 0.0
        else:
            weight = math.comb(upper, train_size - 1) / denominator

        weights.append(weight)

    return np.asarray(weights, dtype=float)


def ccv_1nn_from_profile(
        profile: np.ndarray,
        total_size: int,
        control_size: int,
) -> float:

    weights = _ccv_weights(
        total_size=total_size,
        control_size=control_size
    )

    length = min(len(profile), len(weights))

    return float(np.sum(profile[:length] * weights[:length]))


class GreedyPrototypeSelector:

    def __init__(
            self,
            tolerance: float = 0.0,
            min_per_class: int = 1,
            min_prototypes: int | None = None,
            max_removals: int | None = None,
    ):
        self.tolerance = tolerance
        self.min_per_class = min_per_class
        self.min_prototypes = min_prototypes
        self.max_removals = max_removals

        self.prototype_indices_ = None
        self.history_ = None

    @staticmethod
    def _nearest_two(
            distances: np.ndarray,
            active_mask: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        masked = np.where(
            active_mask[None, :],
            distances,
            np.inf,
        )

        if np.sum(active_mask) == 1:
            nearest = np.argmin(masked, axis=1)
            nearest_dist = masked[
                np.arange(len(masked)),
                nearest,
            ]
            second = nearest.copy()
            second_dist = np.full(
                len(masked),
                np.inf,
            )

            return (
                nearest,
                nearest_dist,
                second,
                second_dist,
            )

        two = np.argpartition(
            masked,
            kth=1,
            axis=1,
        )[:, :2]

        two_dist = np.take_along_axis(
            masked,
            two,
            axis=1,
        )

        sort_order = np.argsort(two_dist, axis=1)

        nearest = np.take_along_axis(
            two,
            sort_order[:, :1],
            axis=1,
        ).ravel()

        second = np.take_along_axis(
            two,
            sort_order[:, 1:2],
            axis=1,
        ).ravel()

        nearest_dist = masked[np.arange(len(masked)), nearest]
        second_dist = masked[np.arange(len(masked)), second]

        return (
            nearest,
            nearest_dist,
            second,
            second_dist,
        )

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> "GreedyPrototypeSelector":

        n_samples = len(x)
        classes = np.unique(y)

        distances = pairwise_euclidean(x, x)
        np.fill_diagonal(distances, np.inf)

        active = np.ones(n_samples, dtype=bool)

        min_prototypes = (
            self.min_prototypes
            if self.min_prototypes is not None
            else len(classes) * self.min_per_class
        )
        min_prototypes = max(
            int(min_prototypes),
            len(classes) * self.min_per_class
        )

        nearest, _, second, _ = self._nearest_two(
            distances,
            active,
        )

        prediction = y[nearest]
        errors = prediction != y
        current_risk = float(np.mean(errors))

        history = [
            {
                "n_prototypes": int(np.sum(active)),
                "loo_risk": current_risk,
            }
        ]

        removed = 0

        while True:
            n_active = int(np.sum(active))

            if n_active <= min_prototypes:
                break

            if (
                self.max_removals is not None
                and removed >= self.max_removals
            ):
                break

            class_counts = {
                class_label: int(np.sum(active & (y == class_label)))
                for class_label in classes
            }

            best_candidate = None
            best_risk = np.inf
            active_indices = np.flatnonzero(active)

            for candidate in active_indices:
                candidate_class = y[candidate]

                if (
                    class_counts[candidate_class]
                    <= self.min_per_class
                ):
                    continue

                affected = nearest == candidate

                if not np.any(affected):
                    candidate_risk = current_risk
                else:
                    new_errors = errors.copy()
                    replacement_labels = y[second[affected]]
                    new_errors[affected] = (replacement_labels != y[affected])
                    candidate_risk = float(np.mean(new_errors))

                if candidate_risk < best_risk:
                    best_risk = candidate_risk
                    best_candidate = candidate

            if best_candidate is None:
                break

            if best_risk > current_risk + self.tolerance:
                break

            active[best_candidate] = False
            removed += 1

            nearest, _, second, _ = self._nearest_two(distances, active)
            prediction = y[nearest]
            errors = prediction != y
            current_risk = float(np.mean(errors))

            history.append(
                {
                    "n_prototypes": int(np.sum(active)),
                    "loo_risk": current_risk,
                }
            )

        self.prototype_indices_ = np.flatnonzero(active)
        self.history_ = history

        return self

    def transform(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        return (
            x[self.prototype_indices_],
            y[self.prototype_indices_],
        )

    def fit_transform(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(x, y)
        return self.transform(x, y)
