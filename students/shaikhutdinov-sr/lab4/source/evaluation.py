from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LinearRegression

from data import PreparedData
from model import PCA


@dataclass
class EvaluationResults:
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    errors: pd.DataFrame
    feature_rmse: np.ndarray
    comparison: pd.DataFrame
    regression_rmse: pd.Series
    baseline_rmse: float


def evaluate_reconstruction(
    model: PCA,
    reference: SklearnPCA,
    train_features: np.ndarray,
    test_features: np.ndarray,
    selected_components: int,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    if not 1 <= selected_components <= len(model.components):
        raise ValueError("Количество компонент должно быть от 1 до числа компонент модели")
    train_coordinates = model.transform(train_features)
    test_coordinates = model.transform(test_features)
    reference_coordinates = reference.transform(test_features)
    rows = []
    reconstruction_differences = []
    feature_rmse = None
    for n_components in range(1, len(model.components) + 1):
        restored_train = model.inverse_transform(train_coordinates[:, :n_components])
        restored_test = model.inverse_transform(test_coordinates[:, :n_components])
        truncated_coordinates = reference_coordinates.copy()
        truncated_coordinates[:, n_components:] = 0
        restored_reference = reference.inverse_transform(truncated_coordinates)
        test_squared_errors = (test_features - restored_test) ** 2
        rows.append({
            "components": n_components,
            "train_mse": np.mean((train_features - restored_train) ** 2),
            "test_mse": test_squared_errors.mean(),
        })
        reconstruction_differences.append(np.max(np.abs(restored_test - restored_reference)))
        if n_components == selected_components:
            feature_rmse = np.sqrt(test_squared_errors.mean(axis=0))
    errors = pd.DataFrame(rows).set_index("components")
    return errors, feature_rmse, max(reconstruction_differences)


def compare_projectors(
    own_components: np.ndarray,
    reference_components: np.ndarray,
) -> float:
    projector_errors = []
    for n_components in range(1, len(own_components) + 1):
        own_basis = own_components[:n_components]
        reference_basis = reference_components[:n_components]
        own_projector = np.dot(own_basis.T, own_basis)
        reference_projector = np.dot(reference_basis.T, reference_basis)
        projector_errors.append(np.linalg.norm(own_projector - reference_projector))
    return max(projector_errors)


def compare_with_reference(
    model: PCA,
    reference: SklearnPCA,
    reconstruction_difference: float,
) -> pd.DataFrame:
    return pd.DataFrame({
        "Проверка": [
            "Сингулярные числа: максимальная разность",
            "Дисперсии компонент: максимальная разность",
            "Матрицы проекции: максимальная норма Фробениуса разности",
            "Восстановленные признаки test: максимальная разность",
        ],
        "Расхождение": [
            np.max(np.abs(model.singular_values - reference.singular_values_)),
            np.max(np.abs(model.explained_variance - reference.explained_variance_)),
            compare_projectors(model.components, reference.components_),
            reconstruction_difference,
        ],
    }).set_index("Проверка")


def evaluate_regression(
    prepared_data: PreparedData,
    model: PCA,
) -> tuple[pd.Series, float]:
    train_coordinates = model.transform(prepared_data.train_features)
    test_coordinates = model.transform(prepared_data.test_features)
    errors = {}
    for n_components in range(1, len(model.components) + 1):
        regression = LinearRegression().fit(
            train_coordinates[:, :n_components], prepared_data.train_targets
        )
        predictions = regression.predict(test_coordinates[:, :n_components])
        errors[n_components] = np.sqrt(np.mean((prepared_data.test_targets - predictions) ** 2))

    baseline = LinearRegression().fit(
        prepared_data.train_features, prepared_data.train_targets
    )
    baseline_predictions = baseline.predict(prepared_data.test_features)
    baseline_rmse = np.sqrt(np.mean((prepared_data.test_targets - baseline_predictions) ** 2))
    return pd.Series(errors, name="test_rmse").rename_axis("components"), float(baseline_rmse)


def evaluate_results(
    prepared_data: PreparedData,
    model: PCA,
    reference: SklearnPCA,
    variance_threshold: float,
) -> EvaluationResults:
    n_components = model.effective_dimension(variance_threshold)
    errors, feature_rmse, reconstruction_difference = evaluate_reconstruction(
        model, reference, prepared_data.train_features, prepared_data.test_features, n_components
    )
    regression_rmse, baseline_rmse = evaluate_regression(prepared_data, model)
    return EvaluationResults(
        n_components=n_components,
        explained_variance_ratio=model.explained_variance_ratio,
        cumulative_variance=np.cumsum(model.explained_variance_ratio),
        errors=errors,
        feature_rmse=feature_rmse,
        comparison=compare_with_reference(model, reference, reconstruction_difference),
        regression_rmse=regression_rmse,
        baseline_rmse=baseline_rmse,
    )
