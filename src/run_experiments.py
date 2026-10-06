"""Run every experiment and save the raw results.

For each instance in INSTANCES, each configuration in CONFIGURATIONS and each
run k = 0 .. N_RUNS-1:
  - seed = BASE_SEED + k
  - run NSGA-II with a fresh random generator seeded with `seed`
  - run SPEA2 with a fresh random generator seeded with the SAME seed
  - record the final front (distinct non-dominated points), runtime (seconds),
    evaluations and generations

Results are written to RAW_DIR/<instance>.json (one file per instance).
An instance that cannot have a feasible solution (see prepare_instance in
src/data_loader.py) is not run; its file records the reason instead.

No metrics are computed here; src/analyze_results.py does that afterwards,
because hypervolume normalisation needs the fronts of ALL runs first.

Run from the project root:   python -m src.run_experiments
"""

import json
import time

import numpy as np

from src.data_loader import InfeasibleInstanceError, get_instance, verify_data_integrity
from src.nsga2 import run_nsga2
from src.objectives import non_dominated
from src.spea2 import run_spea2
from src.util.fetch_config import BASE_SEED, CONFIGURATIONS, INSTANCES, N_RUNS, RAW_DIR

ALGORITHMS = {"NSGA-II": run_nsga2, "SPEA2": run_spea2}


def run_instance(category, name):
    """Run all configurations and runs on one instance; return the record to save."""
    record = {"instance": name, "category": category}
    try:
        inst = get_instance(name)
    except InfeasibleInstanceError as error:
        print(f"\n[{category}] {name}: SKIPPED - {error}")
        record.update(status="infeasible", reason=str(error), runs=[])
        return record

    print(f"\n[{category}] {name}: {inst.m} facilities, {inst.n} customers")
    record.update(status="ok", runs=[])

    for config_name, params in CONFIGURATIONS.items():
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            for algorithm, run_algorithm in ALGORITHMS.items():
                rng = np.random.default_rng(seed)
                start = time.perf_counter()
                result = run_algorithm(inst, params, rng)
                seconds = time.perf_counter() - start
                front = non_dominated(result["front"])     # distinct points, sorted by f1

                record["runs"].append({
                    "algorithm": algorithm,
                    "config": config_name,
                    "run": run,
                    "seed": seed,
                    "time_s": seconds,
                    "evaluations": result["evaluations"],
                    "generations": result["generations"],
                    "front": front.tolist(),                 # list of [f1, f2]
                })
                print(f"  {config_name} run {run + 1:>2}/{N_RUNS}  {algorithm:<7}  "
                      f"{seconds:5.1f} s  {len(front):>3} distinct front points")
    return record


def main():
    verify_data_integrity()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    for category, name in INSTANCES.items():
        record = run_instance(category, name)
        path = RAW_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(record, file)
        print(f"  saved {path}")

    print(f"\nAll experiments finished in {(time.perf_counter() - total_start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()