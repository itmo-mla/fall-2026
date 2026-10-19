from dataclasses import dataclass

import numpy as np

from parzen_knn import ParzenKNN


@dataclass(frozen=True)
class PrototypeSelectionResult:
    prototype_indices: np.ndarray
    noise_indices: np.ndarray
    loo_margins: np.ndarray
    error_history: np.ndarray


def margins_from_scores(scores, labels, classes):
    class_positions = {class_label: index for index, class_label in enumerate(classes)}
    margins = np.empty(len(labels), dtype=float)
    for index, label in enumerate(labels):
        true_position = class_positions[label]
        other_scores = np.delete(scores[index], true_position)
        margins[index] = scores[index, true_position] - other_scores.max()
    return margins


def loo_margins(features, labels, k=1):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    margins = np.empty(len(labels), dtype=float)

    for index in range(len(features)):
        mask = np.arange(len(features)) != index
        model = ParzenKNN(k=k).fit(features[mask], labels[mask])
        scores, _ = model.class_scores(features[index])
        margins[index] = margins_from_scores(scores, labels[index : index + 1], classes)[0]
    return margins


def select_prototypes(features, labels, k=1, noise_threshold=0.0):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    margins = loo_margins(features, labels, k)
    noise_mask = margins <= noise_threshold

    prototypes = []
    for class_label in classes:
        class_indices = np.flatnonzero(labels == class_label)
        best_index = class_indices[np.argmax(margins[class_indices])]
        prototypes.append(int(best_index))
        noise_mask[best_index] = False

    eligible_indices = np.flatnonzero(~noise_mask)
    error_history = []

    while True:
        remaining = np.setdiff1d(eligible_indices, prototypes)
        if len(remaining) == 0:
            error_history.append(0)
            break

        model = ParzenKNN(k=min(k, len(prototypes))).fit(
            features[prototypes], labels[prototypes]
        )
        scores, _ = model.class_scores(features[remaining])
        predictions = model.classes[np.argmax(scores, axis=1)]
        error_mask = predictions != labels[remaining]
        error_history.append(int(error_mask.sum()))
        if not np.any(error_mask):
            break

        error_indices = remaining[error_mask]
        error_scores = scores[error_mask]
        error_margins = margins_from_scores(
            error_scores, labels[error_indices], model.classes
        )
        prototypes.append(int(error_indices[np.argmin(error_margins)]))

    return PrototypeSelectionResult(
        prototype_indices=np.array(prototypes, dtype=int),
        noise_indices=np.flatnonzero(noise_mask),
        loo_margins=margins,
        error_history=np.array(error_history, dtype=int),
    )
