import numpy as np
import pandas as pd


def initial_prototypes(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    prototype_indices = []
    for class_label in np.unique(y_train):
        class_mask = y_train == class_label
        class_indices = np.flatnonzero(class_mask)
        class_features = x_train[class_indices]
        class_center = class_features.mean(axis=0)
        squared_distances = np.sum((class_features - class_center) ** 2, axis=1)

        nearest_position = np.argmin(squared_distances)
        prototype_index = class_indices[nearest_position]
        prototype_indices.append(prototype_index)
    return np.sort(prototype_indices)


def select_prototypes(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    object_count = len(y_train)
    object_indices = np.arange(object_count)
    loo_distances = np.empty((object_count, object_count))
    for i, x in enumerate(x_train):
        loo_distances[i] = np.sqrt(np.sum((x_train - x) ** 2, axis=1))
    np.fill_diagonal(loo_distances, np.inf)

    initial_prototype_indices = initial_prototypes(x_train, y_train)
    prototype_indices = initial_prototype_indices.tolist()
    nearest_prototype_indices = np.empty(object_count, dtype=int)
    nearest_distances = np.empty(object_count)
    for i in object_indices:
        prototype_distances = loo_distances[i, initial_prototype_indices]
        nearest_position = np.argmin(prototype_distances)
        nearest_prototype_indices[i] = initial_prototype_indices[nearest_position]
        nearest_distances[i] = prototype_distances[nearest_position]

    predicted_labels = y_train[nearest_prototype_indices]
    misclassified = predicted_labels != y_train
    current_errors = int(np.count_nonzero(misclassified))
    selection_history = [{
        "prototypes": len(prototype_indices),
        "errors": current_errors,
        "prototype_indices": np.sort(prototype_indices),
    }]

    while len(prototype_indices) < object_count:
        current_predictions = y_train[nearest_prototype_indices]
        candidate_indices = np.setdiff1d(object_indices, prototype_indices)
        best_candidate_index = None
        best_replacement_mask = None
        best_error_count = current_errors

        for candidate_index in candidate_indices:
            candidate_distances = loo_distances[:, candidate_index]
            closer = candidate_distances < nearest_distances
            equal_distances = candidate_distances == nearest_distances
            smaller_index = candidate_index < nearest_prototype_indices
            replacement_mask = closer | (equal_distances & smaller_index)

            predicted_labels = current_predictions.copy()
            predicted_labels[replacement_mask] = y_train[candidate_index]
            misclassified = predicted_labels != y_train
            error_count = int(np.count_nonzero(misclassified))

            if error_count < best_error_count:
                best_candidate_index = int(candidate_index)
                best_error_count = error_count
                best_replacement_mask = replacement_mask

        if best_candidate_index is None:
            break

        best_candidate_distances = loo_distances[:, best_candidate_index]
        nearest_prototype_indices[best_replacement_mask] = best_candidate_index
        nearest_distances[best_replacement_mask] = best_candidate_distances[best_replacement_mask]
        prototype_indices.append(best_candidate_index)
        current_errors = best_error_count
        selection_history.append({
            "prototypes": len(prototype_indices),
            "errors": current_errors,
            "prototype_indices": np.sort(prototype_indices),
        })

    history = pd.DataFrame(selection_history)
    history["risk"] = history["errors"] / object_count
    return np.sort(prototype_indices), history
