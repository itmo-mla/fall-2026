from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import data
import evaluation
import experiments
import reporting


def main() -> None:
    lab_directory = Path(__file__).resolve().parents[1]
    training_parameters = {
        "epochs": 60,
        "regularization": 0.01,
        "momentum": 0.9,
        "quality_weight": 0.001,
        "step_factor": 0.7,
        "seed": 2026,
        "tolerance": 1e-3,
        "patience": 5,
    }
    regularization = training_parameters["regularization"]

    print("Скачивание Wine Quality с Kaggle...")

    prepared_data = data.prepare_data()
    train_data = prepared_data.datasets["train"]

    runs = experiments.run_experiments(
        train_data,
        parameters=training_parameters,
        start_seeds=[11, 23, 37, 51, 79],
    )

    reference = experiments.train_reference(train_data, regularization)

    results = evaluation.evaluate_results(
        prepared_data.datasets, runs, reference, regularization
    )

    sampling_margins = experiments.sample_margin_distributions(
        train_data,
        runs[results.best_run].result.weights,
        sample_count=10 * len(train_data.labels),
        seed=training_parameters["seed"],
    )

    reporting.print_results(prepared_data, results)
    reporting.save_results(
        prepared_data,
        results,
        lab_directory,
        sampling_margins=sampling_margins,
    )
    print("\nГрафики сохранены в images/.")
    print("Таблицы экспериментов и итоговых метрик сохранены в results/.")


if __name__ == "__main__":
    main()
