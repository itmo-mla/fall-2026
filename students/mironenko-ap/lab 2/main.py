"""Точка входа.

Использование:
    python main.py loo       # подбор k методом LOO
    python main.py compare   # сравнение с sklearn
    python main.py select    # отбор эталонов (CNN + ENN)
    python main.py viz       # визуализация
    python main.py all       # всё подряд
"""
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent / "report"
FIGURES = REPORT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def stage_loo():
    from src.loo_search import run_loo_search
    run_loo_search(out_dir=FIGURES)


def stage_compare():
    from src.compare_sklearn import run_compare
    run_compare(out_dir=FIGURES)


def stage_select():
    import pandas as pd
    from src.prototype_selection import run_selection

    loo_csv = FIGURES / "loo_curve.csv"
    if loo_csv.exists():
        df = pd.read_csv(loo_csv).sort_values("loo_error")
        best_err = df["loo_error"].min()
        # при ничьей берём наименьшее k
        k_best = int(df[df["loo_error"] == best_err]["k"].min())
        print(f"LOO-выбор: k = {k_best} (error = {best_err:.4f})")
    else:
        k_best = 3
        print("loo_curve.csv не найден, используем k = 3")

    run_selection(out_dir=FIGURES, k=k_best)


def stage_viz():
    import numpy as np
    from src.plots import run_visualization

    drawn = False
    for tag in ("cnn", "enn", "ccv"):  # +ccv
        proto_path = FIGURES / f"prototypes_{tag}_idx.npy"
        if proto_path.exists():
            idx = np.load(proto_path)
            run_visualization(
                out_dir=FIGURES,
                sample_idx=idx,
                title_suffix=f"with_{tag}_prototypes",
            )
            drawn = True
    if not drawn:
        run_visualization(out_dir=FIGURES, title_suffix="all")


STAGES = {
    "loo": stage_loo,
    "compare": stage_compare,
    "select": stage_select,
    "viz": stage_viz,
}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        for name, fn in STAGES.items():
            print(f"\n===== stage: {name} =====")
            fn()
    elif arg in STAGES:
        STAGES[arg]()
    else:
        print("Неизвестный этап:", arg, "\nДоступные:", list(STAGES) + ["all"])