from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import data
import evaluation
import experiments
import reporting


def main() -> None:
    print("Загрузка HTRU2 с Kaggle...")
    raw_data = data.load_data()
    prepared_data = data.prepare_data(raw_data)
    print(f"Train / test: {len(prepared_data.train.labels)} / {len(prepared_data.test.labels)}")

    fits = experiments.run_experiments(prepared_data.train)
    reference = experiments.train_reference(prepared_data.train)
    results = evaluation.evaluate_results(prepared_data, fits, reference)
    print("\nПроверка эквивалентности:")
    print(results.equivalence.to_string())
    print("\nКачество на тестовой выборке:")
    print(results.metrics.to_string())

    lab_directory = Path(__file__).resolve().parents[1]
    reporting.save_report(prepared_data, results, lab_directory)
    print("\nОтчет сохранен в README.md, графики — в images/.")


if __name__ == "__main__":
    main()
