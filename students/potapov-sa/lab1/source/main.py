from typing import Literal

import numpy as np
import pandas as pd

from src.model import Model
from src.dataset import DataSet, DataSample
from src.linear_classificator import LinearClassificator
from src.plot import plot_loss_history, plot_margins
from src.reference_classifier import ReferenceSGDClassifier
from src.constants import RANDOM_STATE


def solution_corr(
        ds: DataSet,
        plot: bool = False
) -> LinearClassificator:
    model = LinearClassificator(
        ds.feature_count,
        name='corr',
        init_weights_method='corr',
        corr_sample=ds.train
    )
    model.fit(ds)
    if plot:
        plot_margins(model, ds.test, 'corr')
        plot_loss_history(model)
    return model


def solution_multistart(
        ds: DataSet,
        starts: int = 10,
        plot: bool = False
) -> LinearClassificator:
    best_model = None
    for _ in range(starts):
        model = LinearClassificator(
            ds.feature_count,
            name='multistart',
            init_weights_method='random'
        )
        model.fit(ds)
        if (
            best_model is None or
            model.metrics.train.accuracy > best_model.metrics.train.accuracy # type: ignore
        ):
            best_model = model
    if plot:
        plot_margins(best_model, ds.test, 'multistart') # type: ignore
        plot_loss_history(best_model) # type: ignore
    return best_model  # type: ignore


def solution_shuffling(
        ds: DataSet, 
        plot: bool = False
) -> LinearClassificator:
    model = LinearClassificator(
        ds.feature_count,
        name='shuffling',
        object_selection_method='shuffling'
    )
    model.fit(ds)
    if plot:
        plot_margins(model, ds.test, 'shuffling')
        plot_loss_history(model)
    return model


def solution_unsure_margin(
        ds: DataSet, 
        plot: bool = False
) -> LinearClassificator:
    model = LinearClassificator(
        ds.feature_count,
        name='unsure_margin',
        object_selection_method='unsure_first'
    )
    model.fit(ds)
    if plot:
        plot_margins(model, ds.test, 'unsure_margin.png')
        plot_loss_history(model)
    return model


def solution_reference(
        ds: DataSet, 
        plot: bool = False
) -> ReferenceSGDClassifier:
    model = ReferenceSGDClassifier(name='ref')
    model.fit(ds)
    if plot:
        plot_margins(model, ds.test, 'ref')
    return model


def print_model_metrics(model: Model) -> None:
    print(f'{model.name} metrics:')
    print(pd.DataFrame(
        [
            model.metrics.train.to_dict(), # type: ignore
            model.metrics.test.to_dict(), # type: ignore
        ],
        index=['train', 'test']
    ))


def print_models_metrics(*models: Model) -> None:
    index: list[str] = []
    data: list[dict[str, float]] = []
    
    for model in sorted(
        list(models),
        key=lambda m: m.metrics.train.accuracy, # type: ignore
        reverse=True
    ):
        index.append(model.name)
        data.append(model.metrics.train.to_dict()) # type: ignore

    print(pd.DataFrame(data, index=index))


def main():
    N = 100
    print(f'N = {N}')

    best_models: dict[
        Literal[
            'corr', 'multistart', 'shuffling',
            'unsure_margin', 'ref'
        ],
        tuple[Model, DataSample]  # (model, model_ts) ts - test_sample
    ] = {} 
    metrics_data: dict[str, dict[str, list[float]]] = {
        'Accuracy': {},
        'Precision': {},
        'Recall': {}
    }    

    for i in range(1, N + 1):
        ds = DataSet()

        for solution in [ # type: ignore
            solution_corr,
            solution_multistart,
            solution_shuffling,
            solution_unsure_margin,
            solution_reference
        ]:
            model = solution(ds)
            for metric_name, metric_val in model.metrics.train.to_dict().items(): # type: ignore
                if model.name not in metrics_data[metric_name]: # type: ignore
                    metrics_data[metric_name][model.name] = [metric_val] # type: ignore
                else:
                    metrics_data[metric_name][model.name].append(metric_val) # type: ignore
            best_model, _ = best_models.get(model.name) or (None, None) # type: ignore
            if (
                best_model is None or
                model.metrics.train.accuracy > best_model.metrics.train.accuracy # type: ignore
            ):
                best_models[model.name] = (model, ds.test) # type: ignore
        
        print(f'{i}/{N}', end='\r', flush=True)

    print('\n\nBest metrics (train):')
    print_models_metrics(*[model_tuple[0] for model_tuple in best_models.values()])

    best_custom_model, best_custom_model_test_sample = None, None
    for model, test_sample in best_models.values():
        plot_margins(model, test_sample, filename=model.name)
        if model.loss_history_is_loggable:
            plot_loss_history(model)
        if model.name != 'ref' and (
            best_custom_model is None
            or model.metrics.train.accuracy > best_custom_model.metrics.train.accuracy  # type: ignore
        ):
            best_custom_model, best_custom_model_test_sample = model, test_sample
    
    best_custom_model.evaluate(best_custom_model_test_sample)  # type: ignore
    print('\nBest custom model is:', best_custom_model.name)  # type: ignore
    print_model_metrics(best_custom_model)  # type: ignore

    best_ref_model, best_ref_model_test_sample = best_models['ref']
    best_ref_model.evaluate(best_ref_model_test_sample)
    print()
    print_model_metrics(best_ref_model)


if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)
    main()
