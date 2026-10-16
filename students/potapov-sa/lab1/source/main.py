from typing import Literal

import pandas as pd

from src.model import Model
from src.dataset import DataSet, DataSample
from src.linear_classificator import LinearClassificator
from src.plot import plot_margins, plot_metrics
from src.constants import DATASET_TARGET
from src.reference_classifier import ReferenceSGDClassifier


def solution_corr(
        ds: DataSet,
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    model = LinearClassificator(
        ds.feature_count,
        name='corr',
        init_weights_method='corr',
        corr_sample=ds.train
    )
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'corr')
    return model, ds.test


def solution_multistart(
        starts: int = 10,
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    best_model, best_model_ds = None, None
    for _ in range(starts):
        ds = DataSet()
        model = LinearClassificator(
            ds.feature_count,
            name='multistart',
            init_weights_method='random'
        )
        model.fit(ds.train)
        model.evaluate(ds.test)
        if (
            best_model is None or
            model.metrics.test.accuracy > best_model.metrics.test.accuracy # type: ignore
        ):
            best_model, best_model_ds = model, ds
    if plot:
        plot_margins(best_model, best_model_ds.test, 'multistart') # type: ignore
    return best_model, best_model_ds.test # type: ignore


def solution_shuffling(
        ds: DataSet, 
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    model = LinearClassificator(
        ds.feature_count,
        name='shuffling',
        object_selection_method='shuffling'
    )
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'shuffling')
    return model, ds.test


def solution_unsure_margin(
        ds: DataSet, 
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    model = LinearClassificator(
        ds.feature_count,
        name='unsure_margin',
        object_selection_method='unsure_first'
    )
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'unsure_margin.png')
    return model, ds.test


def solution_reference(
        ds: DataSet, 
        plot: bool = False
) -> tuple[ReferenceSGDClassifier, DataSample]:
    model = ReferenceSGDClassifier(name='ref')
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'ref')
    return model, ds.test


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
        key=lambda m: m.metrics.test.accuracy, # type: ignore
        reverse=True
    ):
        index.append(model.name)
        data.append(model.metrics.test.to_dict()) # type: ignore

    print(pd.DataFrame(data, index=index))


def main():
    N = 100
    print(f'N = {N}')

    best_models: dict[
        Literal[
            'corr', 'multistart', 'shuffling',
            'unsure_margin', 'ref'
        ],
        tuple[Model, DataSample]
    ] = {}
    metrics_data: dict[str, dict[str, list[float]]] = {
        'Accuracy': {},
        'Precision': {},
        'Recall': {}
    }

    for i in range(1, N + 1):
        ds = DataSet()

        for model, test_sample in [ # type: ignore
            solution_corr(ds),
            solution_multistart(),
            solution_shuffling(ds),
            solution_unsure_margin(ds),
            solution_reference(ds)
        ]:
            for metric_name, metric_val in model.metrics.test.to_dict().items(): # type: ignore
                if model.name not in metrics_data[metric_name]: # type: ignore
                    metrics_data[metric_name][model.name] = [metric_val] # type: ignore
                else:
                    metrics_data[metric_name][model.name].append(metric_val) # type: ignore
            best_model, best_model_ds = best_models.get(model.name) or (None, None) # type: ignore
            if (
                best_model is None or
                model.metrics.test.accuracy > best_model.metrics.test.accuracy # type: ignore
            ):
                best_models[model.name] = (model, best_model_ds) # type: ignore
        
        print(f'{i}/{N}', end='\r', flush=True)

    print('\n\nBest metrics:')
    print_models_metrics(*[model_tuple[0] for model_tuple in best_models.values()])

    plot_metrics(metrics_data)

    for model, model_ds in best_models.values():
        model.save()
        plot_margins(model, model_ds, filename=model.name)
        print()
        print_model_metrics(model)


def tmp():
    ds = DataSet()
    df = ds.df
    df.info()
    print(df[DATASET_TARGET].value_counts())


if __name__ == '__main__':
    # np.random.seed(RANDOM_STATE)
    # main()
    tmp()
