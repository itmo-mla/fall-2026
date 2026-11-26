import json
import time
from pathlib import Path

import numpy as np
import metric_classifier as mc


def main():
    data = mc.read_dataset(mc.find_dataset())
    grid = mc.RETAINED_FRACTIONS.copy()
    output = Path(__file__).resolve().parent / "experiments"
    output.mkdir(exist_ok=True)
    for filename, seed, fractions in [
        ("coarse_grid.json", 42, [1.0, 0.9, 0.75, 0.5, 0.35, 0.25, 0.1, 0.05, 0.0]),
        ("seed123.json", 123, grid),
    ]:
        start = time.perf_counter()
        mc.RETAINED_FRACTIONS = fractions
        tuning = mc.tune_selection(data, np.arange(len(data["y"])), seed)
        cv = mc.external_cv(data, seed=seed)
        result = {"selection_tuning": tuning, "external_cv": cv,
                  "runtime_seconds": time.perf_counter() - start}
        (output / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    mc.RETAINED_FRACTIONS = grid


if __name__ == "__main__":
    main()
