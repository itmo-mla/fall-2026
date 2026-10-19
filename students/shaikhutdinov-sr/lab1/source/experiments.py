from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeClassifier

from data import Dataset
from model import calculate_margins
from optimization import FitResult, sampling_probabilities, train_sgd


@dataclass
class ExperimentResult:
    initialization: str
    sampling: str
    seed: int
    result: FitResult


def correlation_initialization(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    numerator = np.dot(features.T, labels)
    denominator = np.sum(features ** 2, axis=0)
    weights = np.zeros_like(numerator)
    nonzero_features = denominator > 0
    weights[nonzero_features] = (
        numerator[nonzero_features] / denominator[nonzero_features]
    )
    return weights


def random_initialization(features: np.ndarray, seed: int) -> np.ndarray:
    feature_count = features.shape[1] - 1
    bound = 1 / (2 * feature_count)
    generator = np.random.default_rng(seed)
    return generator.uniform(-bound, bound, features.shape[1])


def run_experiments(
    training_data: Dataset,
    parameters: dict,
    start_seeds: list[int],
    correlation_seed: int = 42,
) -> dict[str, ExperimentResult]:
    runs = {}
    correlation_weights = correlation_initialization(
        training_data.features, training_data.labels
    )
    starts = [("correlation", correlation_seed, correlation_weights)]

    for seed in start_seeds:
        weights = random_initialization(training_data.features, seed)
        starts.append(("random", seed, weights))

    for initialization, seed, initial_weights in starts:
        for sampling in ["uniform", "margin"]:
            name = f"{initialization}_{sampling}_{seed}"
            result = train_sgd(
                training_data,
                initial_weights,
                sampling=sampling,
                **parameters,
            )
            runs[name] = ExperimentResult(initialization, sampling, seed, result)

    steepest_parameters = parameters.copy()
    steepest_parameters["momentum"] = 0.0
    result = train_sgd(
        training_data,
        correlation_weights,
        sampling="uniform",
        step_mode="steepest",
        **steepest_parameters,
    )
    runs["steepest_uniform"] = ExperimentResult(
        initialization="correlation",
        sampling="uniform",
        seed=correlation_seed,
        result=result,
    )
    return runs


def train_reference(
    training_data: Dataset,
    regularization: float,
) -> RidgeClassifier:
    ridge_alpha = len(training_data.labels) * regularization / 2
    reference = RidgeClassifier(
        alpha=ridge_alpha,
        fit_intercept=True,
        solver="cholesky",
    )

    features = training_data.features[:, 1:]
    reference.fit(features, training_data.labels)
    return reference


def sample_margin_distributions(
    training_data: Dataset,
    weights: np.ndarray,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    absolute_margins = np.abs(
        calculate_margins(training_data.features, training_data.labels, weights)
    )
    samples = {}
    for strategy in ["uniform", "margin"]:
        probabilities = sampling_probabilities(
            training_data.features, training_data.labels, weights, strategy
        )
        generator = np.random.default_rng(seed)
        indices = generator.choice(
            len(absolute_margins), size=sample_count, replace=True, p=probabilities
        )
        samples[strategy] = absolute_margins[indices]
    return samples
